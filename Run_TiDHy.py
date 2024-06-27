import hydra
import logging
import numpy as np
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf

import TiDHy.utils.io_dict_to_hdf5 as ioh5
from TiDHy.models.TiDHy import *
from TiDHy.utils.utils import *
from TiDHy.Train_TiDHy import train
from TiDHy.Evaluate_TiDHy import evaluate, fit_SLDS
from TiDHy.datasets import *
from TiDHy.utils.plotting import *

@hydra.main(version_base=None, config_path="conf", config_name="config")
def parse_hydra_config(cfg : DictConfig):
    for k in cfg.paths.keys():
        if k != 'user':
            cfg.paths[k] = Path(cfg.paths[k])
            cfg.paths[k].mkdir(parents=True, exist_ok=True)
    set_logger(cfg,cfg.paths.log_dir/'main.log')

    ##### Set Random Seed #####
    set_seed(cfg.seed)

    ##### Create Tensorboard writer #####   
    train_writer = SummaryWriter(log_dir=cfg.paths.tb_dir / 'train')
    val_writer = SummaryWriter(log_dir=cfg.paths.tb_dir / 'val')
    
    GradNorm_weights=['spatial_rhat','spatial_rbar','temporal']
    if cfg.model['use_r2_decoder']:
        GradNorm_weights.append('r2_losses')
    tb_layout = {'Norms':{
                        'loss_weights': ['Multiline', ['loss_weights/{}'.format(i) for i in GradNorm_weights]],
                        'dyn_norm': ['Multiline', ['dyn_norm/{}'.format(i) for i in range(cfg.model.mix_dim)]],
                        'hyper_norm': ['Multiline', ['hyper_norm/{}'.format(i) for i in range(1)]],
                        'spatial_norm': ['Multiline', ['spatial_norm/{}'.format(i) for i in range(cfg.model.r_dim)]],
                         }}
    train_writer.add_custom_scalars(tb_layout)
    ##### Load Dataset #####
    data_dict, cfg = load_dataset(cfg)
    ##### Convert to float tensors #####
    train_inputs = torch.tensor(data_dict['inputs_train']).float()
    val_inputs = torch.tensor(data_dict['inputs_val']).float()

    input_dim = train_inputs.shape[-1]
    if cfg.train.stack_inputs:
        train_inputs = stack_data(train_inputs,cfg.train.sequence_length,overlap=cfg.train.sequence_length//cfg.train.overlap_factor)
    else:
        train_inputs = train_inputs.reshape(-1, cfg.train.sequence_length, input_dim)
    val_inputs = val_inputs.reshape(-1, cfg.train.sequence_length, input_dim)
    cfg.model.input_dim = input_dim

    logging.info(f'Our inputs have shape: {train_inputs.shape}')
    if cfg.train.batch_size_input:
        batch_size_train = train_inputs.shape[0]
        batch_size_val = val_inputs.shape[0]
    else:
        batch_size_train = cfg.train.batch_size
        # batch_size_val = cfg.train.batch_size
        batch_size_val = val_inputs.shape[0]

    train_dataset = torch.utils.data.TensorDataset(train_inputs)
    val_dataset = torch.utils.data.TensorDataset(val_inputs)
    dataloader_train = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size_train,pin_memory=True,shuffle=True,drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size_val,shuffle=False,pin_memory=True,drop_last=True)
    device = torch.device("cuda:{}".format(cfg.train['gpu']) if torch.cuda.is_available() else "cpu")

    ###### Define the model and optimizer #####
    model = TiDHy(cfg.model, device, show_progress=cfg.train.show_progress, show_inf_progress=cfg.train.show_inf_progress).to(device)
    params_list = []
    params_list.append({'params': list(model.spatial_decoder.parameters()),  'lr': cfg.model.learning_rate_s,'weight_decay': cfg.model.weight_decay})
    if cfg.model.dyn_bias:
        params_list.append({'params': [model.temporal, model.temporal_bias], 'lr': cfg.model.learning_rate_t,'weight_decay': cfg.model.weight_decay})
    else:
        params_list.append({'params': [model.temporal],                      'lr': cfg.model.learning_rate_t,'weight_decay': cfg.model.weight_decay})
    params_list.append({'params': list(model.hypernet.parameters()),         'lr': cfg.model.learning_rate_h,'weight_decay': cfg.model.weight_decay})

    optimizer = [optim.AdamW(params=params_list)]
    milestones = [block for i, block in enumerate(np.arange(cfg.train.num_epochs//5,cfg.train.num_epochs,cfg.train.num_epochs//5))]
    scheduler = [
        MultiStepLR(opt, milestones=milestones, gamma=.5) for opt in optimizer
    ]
    if 'checkpoint' in cfg and cfg['checkpoint'] is not None and cfg['checkpoint'] !='':
        data_load = torch.load(cfg.checkpoint,map_location=device)
        model.load_state_dict(data_load['state_dict'])
        for n,opt in enumerate(optimizer): opt.load_state_dict(data_load['optim_dict'][n])
        for n,sch in enumerate(scheduler): sch.load_state_dict(data_load['sched_dict'][n])
        start_epoch = data_load['epoch']
        best_val_loss = data_load['best_val_loss']
    else:
        start_epoch = 0
        best_val_loss = float("inf")
        
    ##### Save Simulation Params #####
    if cfg.delay.delay_embed:
        reduce_plots = True
    else:
        reduce_plots = False
    logging.info(OmegaConf.to_yaml(cfg))
    OmegaConf.save(cfg, cfg.paths.save_dir / 'config.yaml')
    ##### Start Training #####
    logging.info('Starting Training...')
    
    loss_tr, loss_te = [], []
    with tqdm(initial=start_epoch,total=cfg.train.num_epochs+1, dynamic_ncols=True) as t:
        for epoch in range(cfg.train.num_epochs+1):
            train_metrics, r2_losses_train, _, optimizer = train(model, optimizer, dataloader_train, cfg.model, device)
            
            train_loss = train_metrics['spatial_loss_rhat'] + train_metrics['spatial_loss_rbar'] + train_metrics['temp_loss']
            loss_tr.append(train_loss)
            ##### Write to Tensorboard #####
            train_writer.add_scalar("Total", train_metrics['spatial_loss_rhat'] + train_metrics['spatial_loss_rbar'] + train_metrics['temp_loss'], epoch)
            train_writer.add_scalar("Spatial Loss rhat", train_metrics['spatial_loss_rhat'], epoch)
            train_writer.add_scalar("Spatial Loss rbar", train_metrics['spatial_loss_rbar'], epoch)
            train_writer.add_scalar("Temporal Loss", train_metrics['temp_loss'], epoch)
            dyn_norm = torch.norm(model.temporal.grad.detach().cpu(), p=2, dim=1).numpy()
            hyper_norm = np.mean([torch.norm(param.grad,p=2).detach().cpu().numpy() for param in model.hypernet.parameters()])
            if cfg.model.nonlin_decoder:
                spatial_norm = torch.norm(model.spatial_decoder[-1].weight.grad.detach().cpu(), p=2,dim=0).numpy()
            else:
                spatial_norm = torch.norm(model.spatial_decoder[0].weight.grad.detach().cpu(), p=2,dim=0).numpy()
            if (cfg.model.grad_norm):
                for ind,weight_type in enumerate(GradNorm_weights):
                    train_writer.add_scalar('loss_weights/{}'.format(weight_type), train_metrics['weights'][ind], epoch)
            for ind in range(cfg.model.mix_dim):
                train_writer.add_scalar(f"dyn_norm/{ind}", dyn_norm[ind], epoch)
            train_writer.add_scalar("hyper_norm/0",   hyper_norm,  epoch)
            for ind in range(cfg.model.r_dim):
                train_writer.add_scalar(f"spatial_norm/{ind}", spatial_norm[ind], epoch)
            if cfg.model.nonlin_decoder:
                fig, ax = plot_spatial_rf(model.spatial_decoder[-1].weight.data.detach().cpu().numpy().T)
            else:
                fig, ax = plot_spatial_rf(model.spatial_decoder[0].weight.data.detach().cpu().numpy().T)
            train_writer.add_figure("RF", fig, epoch)
            if cfg.model.low_rank_temp:
                V = torch.bmm(model.temporal.unsqueeze(-1),model.temporal.unsqueeze(1)).data.cpu().detach().numpy()
                fig, ax = plot_dynamic_matrix(V)
            else:
                fig, ax = plot_dynamic_matrix(model.temporal.data.cpu().detach().numpy().reshape(cfg.model.mix_dim,cfg.model.r_dim,cfg.model.r_dim))
            train_writer.add_figure("Dynamics", fig, epoch)

            # adjust learning rate
            for n,sch in enumerate(scheduler): sch.step()
            
            if epoch % cfg.train.save_summary_steps == 0:
                val_metrics, loss_avg, result_dict = evaluate(model, dataloader_val, cfg.model, device)

                val_loss = val_metrics['spatial_loss_rhat'] + val_metrics['spatial_loss_rbar'] + val_metrics['temp_loss']
                loss_te.append(val_loss)
                is_best = val_loss <= best_val_loss
                if is_best:
                    best_val_loss = val_loss
                # Save weights
                model_dict = {key:value for key,value in model.__dict__.items() if key[0] != '_'}
                save_checkpoint({'epoch': epoch + 1,
                                 'best_val_loss': best_val_loss,
                                 'state_dict': model.state_dict(),
                                 'optim_dict': [opt.state_dict() for opt in optimizer],
                                 'sched_dict': [sch.state_dict() for sch in scheduler],
                                 'model_dict': model_dict},
                                 is_best=is_best,
                                 checkpoint=cfg.paths.ckpt_dir,epoch=epoch)
                ##### Write Figs to TB #####
                nfig=0
                for key in result_dict.keys():
                    result_dict[key] = result_dict[key].cpu().detach().numpy()
                
                if 'LDS' in cfg.dataset_name:
                    result_dict['As'] = data_dict['As']
                    result_dict['bs'] = data_dict['bs']
                    result_dict['states_x_val'] = data_dict['states_x_val']

                nfig,fig,axs = plot_inputs(nfig,result_dict,cfg,figsize=(10,6))
                train_writer.add_figure("inputs", fig, epoch)
                if (cfg.dataset_name == 'LDS'):
                    nfig,figs,axs = plot_latent_LDS(nfig,result_dict,cfg)
                    for n,fig in enumerate(figs):
                        train_writer.add_figure(f"latents{n}", fig, epoch)
                elif (cfg.dataset_name=='SLDS'): 
                    nfig,figs,axs = plot_latent_LDS(nfig,result_dict,cfg) 
                    for n,fig in enumerate(figs):
                        train_writer.add_figure(f"latents{n}", fig, epoch)
                    nfig,figs,axs = plot_SLDS_plots(nfig,result_dict,data_dict,cfg) 
                    for n,fig in enumerate(figs):
                        train_writer.add_figure(f"SLDS_fig{n}", fig, epoch)
                else:
                    nfig,figs,axs = plot_latent_states(nfig,result_dict,cfg)
                    train_writer.add_figure(f"latents", figs, epoch)
                nfig,fig,axs = plot_Ws(nfig,result_dict,cfg)
                train_writer.add_figure("W", fig, epoch)
                nfig,fig,axs = plot_R2_hat(nfig,result_dict,cfg)
                train_writer.add_figure("R2", fig, epoch)

                ##### Write to Tensorboard #####
                val_writer.add_scalar("Total", val_loss, epoch)
                val_writer.add_scalar("Spatial Loss rhat", val_metrics['spatial_loss_rhat'], epoch)
                val_writer.add_scalar("Spatial Loss rbar", val_metrics['spatial_loss_rbar'], epoch)
                val_writer.add_scalar("Temporal Loss", val_metrics['temp_loss'], epoch)

            ##### Update tqdm #####
            t.set_postfix(val_loss='{:05.3f}'.format(val_loss),train_loss='{:05.3f}'.format(train_loss))
            t.update()
            ##### Logging Losses #####
            logging.info('Epoch:{}, Training Loss:{}, Test Loss:{}'.format(epoch,train_loss,val_loss))

    ##### Load Best Model #####
    torch.cuda.empty_cache()
    set_seed(42)
    data_load = torch.load(cfg.paths.ckpt_dir/'best.pth.tar',map_location=device)
    model.load_state_dict(data_load['state_dict'])
    logging.info('Loaded from {} epoch'.format(data_load['epoch']))
    ##### Load Test Dataset #####
    test_inputs = torch.tensor(data_dict['inputs_test']).float()
    test_inputs = test_inputs.reshape(-1, cfg.train.sequence_length, input_dim)
    if (cfg.train.batch_size_input) | (test_inputs.shape[0] < cfg.train.batch_size):
        batch_size_test = test_inputs.shape[0]
    else:
        batch_size_test = cfg.train.batch_size
    test_dataset = torch.utils.data.TensorDataset(test_inputs)
    dataloader_test = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size_test,pin_memory=True,shuffle=False,drop_last=True)
    for Nbatch, batch in enumerate(dataloader_test):
        X = batch[0].to(device,non_blocking=True)
        spatial_loss_rhat, spatial_loss_rbar,temp_loss,result_dict_temp = model.evaluate_record(X)
        if Nbatch==0:
            result_dict = result_dict_temp
        else:
            for key in result_dict.keys():
                if isinstance(result_dict[key],torch.Tensor):
                    result_dict[key] = torch.cat((result_dict[key],result_dict_temp[key]),dim=0)
    # spatial_loss, temp_loss, result_dict = model.evaluate_record(X)

    for key in result_dict.keys():
        result_dict[key] = result_dict[key].cpu().detach().numpy()
    
    if (cfg.dataset_name != 'CalMS21') & (cfg.dataset_name != 'Bowen') & (cfg.dataset_name != 'AnymalTerrain') & (cfg.dataset_name != 'MABe2022'):
        if (cfg.dataset_name != 'Lorenz'):
            result_dict['As'] = data_dict['As']
            result_dict['bs'] = data_dict['bs']
        result_dict['states_x_test'] = data_dict['states_x_test']
    result_dict['loss_tr'] = np.array(loss_tr)
    result_dict['loss_te'] = np.array(loss_te)
    result_dict['final_spatial_loss_rhat'] = spatial_loss_rhat.item()
    result_dict['final_spatial_loss_rbar'] = spatial_loss_rbar.item()
    result_dict['final_temp_loss'] = temp_loss.item()
    for key in result_dict.keys():
        if isinstance(result_dict[key],dict):
            result_dict[key] = [result_dict[key][key2] for key2 in result_dict[key].keys()]
    ##### Save Results #####
    ioh5.save(cfg.paths.log_dir/'results.h5',result_dict)
    
    if (cfg.dataset_name == 'SLDS'):
        fit_SLDS(cfg,data_dict)
    
if __name__ == "__main__":
    parse_hydra_config()
    