import numpy as np
import torch
import torch.nn.functional as F

from TiDHy.utils import RunningAverage, cos_sim_mat

def train(model, optimizer, dataloader, params, device):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = RunningAverage()
    loss_weights = []
    # Use tqdm for progress bar
    # with tqdm(total=len(dataloader), dynamic_ncols=True) as t:
    for batch_num, train_batch in enumerate(dataloader):
        # move to GPU if available
        if ('cuda' in device.type) | ('mps' in device.type):
            train_batch = train_batch[0].to(device,non_blocking=True)
        else: 
            train_batch = train_batch[0]
        ##### Forward pass #####
        spatial_loss_rhat, spatial_loss_rbar, temp_loss, r2_loss, _, _ = model(train_batch)
        # compute loss
        loss_dict = {
            "spatial_loss_rhat": spatial_loss_rhat.item(),
            "spatial_loss_rbar": spatial_loss_rbar.item(),
            "temp_loss": temp_loss.item(),
        }
        # clear previous gradients, compute gradients of all variables wrt loss
        for opt in optimizer: opt.zero_grad()
        ##### Regularization #####
        cos_reg = 1/model.mix_dim * cos_sim_mat(model.temporal)
        if 'Lnuc_alpha' in params and params['Lnuc_alpha'] is not None and params['Lnuc_alpha'] !=''  and params['Lnuc_alpha'] != 0:
            Lnuc_sparcity_reg = torch.sum(torch.norm(model.temporal.reshape(-1,model.r_dim,model.r_dim), p='nuc',dim=(-2,-1)))
        if 'L0_alpha' in params and params['L0_alpha'] is not None and params['L0_alpha'] !=''  and params['L0_alpha'] != 0:
            L0_sparcity_reg = torch.sum(torch.norm(model.temporal, p=0, dim=-1))
        if 'L1_alpha' in params and params['L1_alpha'] is not None and params['L1_alpha'] !=''  and params['L1_alpha'] != 0:
            sparcity_reg = torch.sum(torch.norm(model.temporal, p=1, dim=-1))
        if 'L1_alpha_W' in params and params['L1_alpha_W'] is not None and params['L1_alpha_W'] !='' and params['L1_alpha_W'] != 0:
            sparcity_W = torch.norm(model.hypernet[-2].weight, p=1)
        if 'L1_alpha_spat' in params and params['L1_alpha_spat'] is not None and params['L1_alpha_spat'] !='' and params['L1_alpha_spat'] != 0:
            sparcity_spat = torch.sum(torch.norm(model.spatial_decoder[0].weight, p=1, dim=1))
        if 'Orth_alpha_spat' in params and params['Orth_alpha_spat'] is not None and params['Orth_alpha_spat'] !='' and params['Orth_alpha_spat'] != 0:
            Orth_spat = torch.pow((model.spatial_decoder[0].weight.T @ model.spatial_decoder[0].weight) - (torch.eye(model.spatial_decoder[0].weight.shape[1],device=model.spatial_decoder[0].weight.device)),2).mean()
        if 'Orth_alpha_r2' in params and params['Orth_alpha_r2'] is not None and params['Orth_alpha_r2'] !='' and params['Orth_alpha_r2'] != 0:
            Orth_r2 = 1/model.r2_dim * cos_sim_mat(model.hypernet[0].weight.T)
        ##### GradNorm #####
        if (params.grad_norm):
            losses = [spatial_loss_rhat, spatial_loss_rbar, temp_loss+params.cos_eta*cos_reg]
            if 'Lnuc_alpha' in params and params['Lnuc_alpha'] is not None and params['Lnuc_alpha'] !=''  and params['Lnuc_alpha'] != 0:
                for n in range(len(losses)):
                    losses[n] += params.Lnuc_alpha*Lnuc_sparcity_reg 
            if 'L0_alpha' in params and params['L0_alpha'] is not None and params['L0_alpha'] !=''  and params['L0_alpha'] != 0:
                for n in range(len(losses)):
                    losses[n] += params.L1_alpha*L0_sparcity_reg 
            if 'L1_alpha' in params and params['L1_alpha'] is not None and params['L1_alpha'] !=''  and params['L1_alpha'] != 0:
                for n in range(len(losses)):
                    losses[n] += params.L1_alpha*sparcity_reg
            if 'L1_alpha_spat' in params and params['L1_alpha_spat'] is not None and params['L1_alpha_spat'] !='' and params['L1_alpha_spat'] != 0:
                for n in range(len(losses)):
                    losses[n] += params.L1_alpha_spat*sparcity_spat
            if 'Orth_alpha_spat' in params and params['Orth_alpha_spat'] is not None and params['Orth_alpha_spat'] !='' and params['Orth_alpha_spat'] != 0:
                for n in range(len(losses)):
                    losses[n] += params.Orth_alpha_spat*Orth_spat
            if 'Orth_alpha_r2' in params and params['Orth_alpha_r2'] is not None and params['Orth_alpha_r2'] !='' and params['Orth_alpha_r2'] != 0:
                losses[-1] += params.Orth_alpha_r2*Orth_r2
            if params.use_r2_decoder:
                losses.append(r2_loss)
            ####### Take Gradient Step ######
            weights, loss, optimizer = grad_norm(model,params,torch.stack(losses),optimizer)
        else:
            loss = spatial_loss_rhat + spatial_loss_rbar + temp_loss + params.cos_eta*cos_reg 
            if 'Lnuc_alpha' in params and params['Lnuc_alpha'] is not None and params['Lnuc_alpha'] !=''  and params['Lnuc_alpha'] != 0:
                loss += params.Lnuc_alpha*Lnuc_sparcity_reg 
            if 'L0_alpha' in params and params['L0_alpha'] is not None and params['L0_alpha'] !=''  and params['L0_alpha'] != 0:
                loss += params.L1_alpha*L0_sparcity_reg 
            if 'L1_alpha' in params and params['L1_alpha'] is not None and params['L1_alpha'] !=''  and params['L1_alpha'] != 0:
                loss += params.L1_alpha*sparcity_reg 
            if 'L1_alpha_spat' in params and params['L1_alpha_spat'] is not None and params['L1_alpha_spat'] !='' and params['L1_alpha_spat'] != 0:
                loss += params.L1_alpha_spat*sparcity_spat
            if 'L1_alpha_W' in params and params['L1_alpha_W'] is not None and params['L1_alpha_W'] !='' and params['L1_alpha_W'] != 0:
                loss += params.L1_alpha_W*sparcity_W
            if 'Orth_alpha_spat' in params and params['Orth_alpha_spat'] is not None and params['Orth_alpha_spat'] !='' and params['Orth_alpha_spat'] != 0:
                loss += params.Orth_alpha_spat*Orth_spat
            if 'Orth_alpha_r2' in params and params['Orth_alpha_r2'] is not None and params['Orth_alpha_r2'] !='' and params['Orth_alpha_r2'] != 0:
                loss += params.Orth_alpha_r2*Orth_r2
            if params.use_r2_decoder:
                losses += r2_loss
            loss.backward()

            ##### Clip gradients if too large #####
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip_grad)
            # performs updates using calculated gradients
            for opt in optimizer: opt.step()
        
        # ###### Clip weights to zero ######
        if 'L0_alpha' in params and params['L0_alpha'] is not None and params['L0_alpha'] !=''  and params['L0_alpha'] != 0:
            with torch.no_grad():
                close_to_zero = torch.abs(model.temporal)<1e-4
                model.temporal[close_to_zero] = 0

        # normalize
        if (params.normalize_temporal) | (params.normalize_spatial):
            model.normalize()
        # compute all metrics on this batch
        summ.append(loss_dict)

        # update the average loss
        loss_avg.update(loss.item())
        torch.cuda.empty_cache()


    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    if (params.grad_norm):
        metrics_mean['weights'] = weights.detach().cpu().numpy()
        metrics_mean['weighted_loss'] = loss.detach().cpu().numpy()

    return metrics_mean, r2_loss, loss_avg, optimizer

def grad_norm(model,params,loss,optimizer):

    if model.iters == 0:
        # init weights
        weights = torch.ones_like(loss)
        weights = torch.nn.Parameter(weights)
        T = weights.sum().detach() # sum of weights
        # set optimizer for weights
        optimizer.append(torch.optim.AdamW([weights], lr=params.lr_weights, weight_decay=params.weight_decay))
        # set L(0)
        model.l0 = loss.detach()
    else:
        weights = model.loss_weights
        T = weights.sum().detach() # sum of weights
        weights = torch.nn.Parameter(weights)
        optimizer[-1] = torch.optim.AdamW([weights], lr=params.lr_weights, weight_decay=params.weight_decay)
    # compute the weighted loss
    weighted_loss = weights @ loss
    # clear gradients of network
    for opt in optimizer[:-1]: opt.zero_grad()
    # backward pass for weigthted task loss
    weighted_loss.backward(retain_graph=True)

    # compute the L2 norm of the gradients for each task
    gw = []
    ##### Spatial, Temporal  #####
    model_parameters = [[p for name,p in model.spatial_decoder.named_parameters()],
                        [p for name,p in model.spatial_decoder.named_parameters()]+[model.temporal],
                        [model.temporal]+[p for name,p in model.hypernet.named_parameters()],
                        ]
    if 'L0_alpha' in params and params['L0_alpha'] is not None and params['L0_alpha'] !=''  and params['L0_alpha'] != 0:
        model_parameters.append(model.temporal)
    if 'L1_alpha' in params and params['L1_alpha'] is not None and params['L1_alpha'] !=''  and params['L1_alpha'] != 0:
        model_parameters.append(model.temporal)
    if 'L1_alpha_spat' in params and params['L1_alpha_spat'] is not None and params['L1_alpha_spat'] !='' and params['L1_alpha_spat'] != 0:
        model_parameters.append([p for name,p in model.spatial_decoder.named_parameters()])
    if params.use_r2_decoder:
        model_parameters.append([p for name,p in model.hypernet.named_parameters()])
    for i in range(len(loss)):
        dl = torch.autograd.grad(weights[i]*loss[i], model_parameters[i], retain_graph=True, create_graph=True)
        if len(dl)>1:
            dl_norm = 0
            for n in range(len(dl)):
                dl_norm += torch.norm(dl[n])
        else:
            dl_norm = torch.norm(dl[0])
        gw.append(dl_norm)
    gw = torch.stack(gw)
    # compute loss ratio per task
    loss_ratio = loss.detach() / model.l0
    # compute the relative inverse training rate per task
    rt = loss_ratio / loss_ratio.mean()
    # compute the average gradient norm
    gw_avg = gw.mean().detach()
    # compute the GradNorm loss
    constant = (gw_avg * rt ** params.grad_alpha).detach()
    gradnorm_loss = torch.abs(gw - constant).sum()
    
    # backward pass for GradNorm
    optimizer[-1].zero_grad()
    gradnorm_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip_grad)
    # update model weights and loss weights
    for opt in optimizer: opt.step()
    # renormalize weights
    weights = torch.exp(weights)
    weights = (weights / weights.sum() * T).detach()
    model.loss_weights = weights.detach()
    
    # update iters
    model.iters += 1
    return weights.detach(), weighted_loss.detach(), optimizer
