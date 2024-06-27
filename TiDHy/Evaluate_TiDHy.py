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
        spatial_loss_rhat, spatial_loss_rbar, temp_loss, result_dict = model.evaluate_record(data_batch)
        # compute loss
        loss_dict = {
            "spatial_loss_rhat": spatial_loss_rhat.item(),
            "spatial_loss_rbar": spatial_loss_rbar.item(),
            "temp_loss": temp_loss.item(),
        }
        # compute all metrics on this batch
        summ.append(loss_dict)
        cos_reg = torch.sum(torch.abs(torch.tril(F.normalize(model.temporal.reshape(model.mix_dim, -1)) @ F.normalize(model.temporal.reshape(model.mix_dim, -1)).t(),diagonal=-1)))
        loss = spatial_loss_rhat + spatial_loss_rbar + temp_loss + params.cos_eta*cos_reg
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
    N = inputs_train_SLDS.shape[-1]
    K = len(np.unique(full_state_z))
    D = data_dict['states_x_test'].shape[-1]
    print(f"N:{N} K: {K}, D: {D}")
    # Create the model and initialize its parameters
    slds = ssm.SLDS(N=N, #inputs_train_SLDS.shape[-1], # Input dimension
                    K=K, #len(np.unique(full_state_z)), # number of sets of dynamics dimensions
                    D=D, #data_dict['states_x_test'].shape[-1], # latent dim
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

    with open(cfg.paths.log_dir/f'ssm_slds_test_full_{D}D_{KeyboardInterrupt}K_{cfg.dataset.ssm_params.seed}seed.pickle', 'wb') as handle:
        pickle.dump(slds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(posterior, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(q_lem_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(q_lem_z, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(q_lem_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ###### rSLDS
    print('Fitting rSLDS with Laplace-EM')
    rslds = ssm.SLDS(N=N, #inputs_train_SLDS.shape[-1], # Input dimension
                     K=K, #len(np.unique(full_state_z)), # number of sets of dynamics dimensions
                     D=D, #data_dict['states_x_test'].shape[-1], # latent dim
                     transitions="recurrent",
                     emissions="gaussian")
    
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
    
    with open(cfg.paths.log_dir/f'ssm_rslds_test_full_{D}D_{K}K_{cfg.dataset.ssm_params.seed}seed.pickle', 'wb') as handle:
        pickle.dump(rslds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(posterior, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(q_lem_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(q_lem_z, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(q_lem_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
