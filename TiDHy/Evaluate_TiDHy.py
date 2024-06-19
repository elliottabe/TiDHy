import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from TiDHy.utils import RunningAverage 

def evaluate(model, dataloader, params, device):
    """Evaluate the model on `dataloader` batches.

    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        params: (Params) hyperparameters
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    loss_avg = RunningAverage()

    # compute metrics over the dataset
    for data_batch in dataloader:

        # move to GPU if available
        if ('cuda' in device.type) | ('mps' in device.type):
            data_batch = data_batch[0].to(device,non_blocking=True)
        else: 
            train_batch = data_batch[0]
        spatial_loss, temp_loss, result_dict = model.evaluate_record(data_batch)
        # compute loss
        loss_dict = {
            "spatial_loss": spatial_loss.item(),
            "temp_loss": temp_loss.item(),
        }
        # compute all metrics on this batch
        summ.append(loss_dict)
        cos_reg = torch.sum(torch.abs(torch.tril(F.normalize(model.temporal.reshape(model.mix_dim, -1)) @ F.normalize(model.temporal.reshape(model.mix_dim, -1)).t(),diagonal=-1)))
        loss = spatial_loss + temp_loss + params.cos_eta*cos_reg
        # update the average loss
        loss_avg.update(loss.item())

        # t.set_postfix(eval_loss='{:05.3f}'.format(loss_avg()))
        # t.update()

    ##### Reset R2 State after eval #####
    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean(np.mean([x[metric] for x in summ])) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    # logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean, loss_avg, result_dict


def record(model, data_batch, turnoff=None):
    """Record model predictions and inferences on `data_batch`

    Args:
        model (torch.nn.Module): the neural network
        data_batch (torch.Tensor): a batch of data with dimension `batch x time x feature`
        turnoff (int, optional): the time step to stop the input and let the model predict the rest of steps. Must be no larger than `data_batch.shape[1]`. 
                                 Defaults to None (use all time steps).

    Returns:
        dict: a dictionary of tensors
    """
    model.eval()
    batch_size = data_batch.size(0)
    T = data_batch.size(1)
    if turnoff is None:
        turnoff = T
    assert turnoff <= T, "Input turnoff time larger than the total sequence length"
    input_dim = data_batch.size(2)
    # saving values
    I_bar = torch.zeros((batch_size, T, input_dim))             # Image prediction from hypernet
    I_hat = torch.zeros((batch_size, T, input_dim))             # Image correction (turned off after turnoff)
    I = torch.zeros((batch_size, T, input_dim))                 # True input
    R_bar = torch.zeros((batch_size, T, model.r_dim))           # prediction from hypernet
    R_hat = torch.zeros((batch_size, T, model.r_dim))           # ISTA correction (turned off after turnoff)
    R2_hat = torch.zeros((batch_size, T, model.r2_dim))         # Embedding (same after turnoff)
    W = torch.zeros((batch_size, T, model.mix_dim))             # Mixture weights
    if model.dyn_bias:
        b = torch.zeros((batch_size, T, model.mix_dim))             # Bias
    Vt = torch.zeros((batch_size, T, model.r_dim, model.r_dim)) # Temporal prediction matrices
    # initialize embedding
    r, r2 = model.init_code_(batch_size)
    R_bar[:, 0] = r.clone().detach().cpu()
    R2_hat[:, 0] = r2.clone().detach().cpu()
    I_bar[:, 0] = model.spatial_decoder(r).clone().detach().cpu()
    # p(r_1 | I_1)
    r = model.inf_first_step(data_batch[:, 0])
    R_hat[:, 0] = r.clone().detach().cpu()
    I_hat[:, 0] = model.spatial_decoder(r).clone().detach().cpu()
    I[:, 0] = data_batch[:, 0]
    # input on
    for t in tqdm(range(1, turnoff), leave=False):
        r_p = r.clone().detach()
        # hypernet prediction
        r_bar,V_t,w = model.temporal_prediction_(r_p, r2)
        R_bar[:, t] = r_bar.clone().detach().cpu()
        I_bar[:, t] = model.spatial_decoder(r_bar).clone().detach().cpu()
        # inference
        r, r2, _ = model.inf(data_batch[:, t], r_p, r2.clone().detach())
        R_hat[:, t] = r.clone().detach().cpu()
        I_hat[:, t] = model.spatial_decoder(r).clone().detach().cpu()
        I[:, t] = data_batch[:, t]
        R2_hat[:, t]= r2.clone().detach().cpu()
        wb = model.hypernet(r2)
        W[:, t] = wb[:,:model.mix_dim].reshape(batch_size, -1).clone().detach().cpu()
        if model.dyn_bias:
            b[:, t] = wb[:,model.mix_dim:].reshape(batch_size, -1).clone().detach().cpu()
        Vt[:, t] = V_t.clone().detach().cpu()
    # input off, no more inference
    for t in range(turnoff, T):
        # predict
        r_bar,V_t,w = model.temporal_prediction_(r, r2)
        #r_hat = model.prediction_(r, r2)
        R_bar[:, t] = r_bar.clone().detach().cpu()
        I_bar[:, t] = model.spatial_decoder(r_bar).clone().detach().cpu()
        # no more correction
        R_hat[:, t] = R_bar[:, t]
        I_hat[:, t] = I_bar[:, t]
        # no more fitting embedding
        R2_hat[:, t] = R2_hat[:, t-1]
        I[:, t] = data_batch[:, t]
        W[:, t] = W[:, t-1]
        r = r_bar
        # result dict
    result_dict = {}
    result_dict['I_bar'] = I_bar
    result_dict['I_hat'] = I_hat
    result_dict['I'] = I
    result_dict['R_bar'] = R_bar
    result_dict['R_hat'] = R_hat
    result_dict['R2_hat'] = R2_hat
    result_dict['W'] = W
    result_dict['Vt'] = Vt
    if model.dyn_bias:
        result_dict['b'] = b
        
    return result_dict


def fit_SLDS(cfg,data_dict):
    import ssm, itertools, pickle 
    from ssm.util import find_permutation
    print("Fitting SLDS with Laplace-EM")
    inputs_train_SLDS= data_dict['inputs_train']
    inputs_test_SLDS=data_dict['inputs_test'] 

    states_x_test = data_dict['states_x_test']
    states_z_test = data_dict['states_z_test']
    # states_z_test = data_dict['states_z']
    ssm_params = cfg.dataset.ssm_params
    ##### Set up combinatorics of timescales #####
    lst = list(itertools.product([0, 1], repeat=3))
    lst2 = list(itertools.product(['F', 'S'], repeat=3))
    full_state_z = np.zeros(ssm_params['time_bins_test'],dtype=int)
    for n in range(len(lst)):
        full_state_z[np.apply_along_axis(lambda x: np.all(x == lst[n]),0,states_z_test)] = n
        
    # Create the model and initialize its parameters
    slds = ssm.SLDS(N=inputs_train_SLDS.shape[-1], # Input dimension
                    K=len(np.unique(full_state_z)), # number of sets of dynamics dimensions
                    D=data_dict['states_x_test'].shape[-1], # latent dim
                    emissions="gaussian")

    # Fit the model using Laplace-EM with a structured variational posterior
    q_lem_elbos, q_lem = slds.fit(inputs_train_SLDS, method="laplace_em",
                                variational_posterior="structured_meanfield",
                                initialize=False,
                                num_iters=100, alpha=0.0,)


    posterior = slds._make_variational_posterior( variational_posterior="structured_meanfield",datas=inputs_test_SLDS,inputs=None, masks=None, tags=None,method="laplace_em")
    q_lem_x = posterior.mean_continuous_states[0]

    # # Find the permutation that matches the true and inferred states
    slds.permute(find_permutation(full_state_z, slds.most_likely_states(q_lem_x, inputs_test_SLDS)))
    q_lem_z = slds.most_likely_states(q_lem_x, inputs_test_SLDS)

    # Smooth the data under the variational posterior
    q_lem_y = slds.smooth(q_lem_x, inputs_test_SLDS)

    with open(cfg.paths.log_dir/f'ssm_slds_test_full_{cfg.model.r_dim}D_{cfg.model.r2_dim}K_{cfg.dataset.ssm_params.seed}seed.pickle', 'wb') as handle:
        pickle.dump(slds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(posterior, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(q_lem_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(q_lem_z, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(q_lem_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ###### rSLDS
    print('Fitting rSLDS with Laplace-EM')
    rslds = ssm.SLDS(N=inputs_train_SLDS.shape[-1], # Input dimension
                    K=len(np.unique(full_state_z)), # number of sets of dynamics dimensions
                    D=data_dict['states_x_test'].shape[-1], # latent dim
                    transitions="recurrent",
                    emissions="gaussian",
                    single_subspace=True)
    
    # rslds.initialize(inputs_train_SLDS)
    q_elbos_lem, q_lem = rslds.fit(inputs_train_SLDS, method="laplace_em",
                                variational_posterior="structured_meanfield",
                                initialize=False, num_iters=100, alpha=0.0)
    # xhat_lem = q_lem.mean_continuous_states[0]
    # rslds.permute(find_permutation(full_state_z, rslds.most_likely_states(xhat_lem, inputs_test_SLDS)))
    # zhat_lem = rslds.most_likely_states(xhat_lem, inputs_test_SLDS)

    posterior = rslds._make_variational_posterior( variational_posterior="structured_meanfield",datas=inputs_test_SLDS,inputs=None, masks=None, tags=None,method="laplace_em")
    q_lem_x = posterior.mean_continuous_states[0]

    # # Find the permutation that matches the true and inferred states
    rslds.permute(find_permutation(full_state_z, rslds.most_likely_states(q_lem_x, inputs_test_SLDS)))
    q_lem_z = rslds.most_likely_states(q_lem_x, inputs_test_SLDS)

    # Smooth the data under the variational posterior
    q_lem_y = rslds.smooth(q_lem_x, inputs_test_SLDS)
    
    with open(cfg.paths.log_dir/f'ssm_rslds_test_full_{cfg.model.r_dim}D_{cfg.model.r2_dim}K_{cfg.dataset.ssm_params.seed}seed.pickle', 'wb') as handle:
        pickle.dump(rslds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(posterior, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(q_lem_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(q_lem_z, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(q_lem_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
