import hydra
import logging
import numpy as np
import itertools
import ssm
from ssm.util import random_rotation, find_permutation
import pickle

from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf

import TiDHy.utils.io_dict_to_hdf5 as ioh5
from TiDHy.utils import *
from TiDHy.datasets import *
from TiDHy.Evaluate_TiDHy import fit_SLDS

@hydra.main(version_base=None, config_path="conf", config_name="config")
def parse_hydra_config(cfg : DictConfig):
    for k in cfg.paths.keys():
        if k != 'user':
            cfg.paths[k] = Path(cfg.paths[k])
            cfg.paths[k].mkdir(parents=True, exist_ok=True)
    set_logger(cfg,cfg.paths.log_dir/'main.log')

    ##### Set Random Seed #####
    set_seed(42)

    
    # ##### Load Dataset #####
    data_dict, cfg = load_dataset(cfg)
    
    fit_SLDS(cfg,data_dict)
    # print("Fitting SLDS with Laplace-EM")
    # inputs_train_SLDS= data_dict['inputs_train']#/np.max(np.abs(data_dict['inputs_train']),axis=0)
    # inputs_test_SLDS=data_dict['inputs_test'] # (data_dict['inputs_test']-np.mean(data_dict['inputs_test'],axis=-1,keepdims=True))/np.std(data_dict['inputs_test'],axis=-1,keepdims=True)

    # states_x_test = data_dict['states_x_test']
    # states_z_test = data_dict['states_z_test']
    # # states_z_test = data_dict['states_z']
    # ssm_params = cfg.dataset.ssm_params
    # ##### Set up combinatorics of timescales #####
    # lst = list(itertools.product([0, 1], repeat=3))
    # lst2 = list(itertools.product(['F', 'S'], repeat=3))
    # full_state_z = np.zeros(ssm_params['time_bins_test'],dtype=int)
    # # full_state_z = np.zeros(ssm_params['time_bins_train'],dtype=int)
    # for n in range(len(lst)):
    #     full_state_z[np.apply_along_axis(lambda x: np.all(x == lst[n]),0,states_z_test)] = n
        
    # # Create the model and initialize its parameters
    # slds = ssm.SLDS(N=inputs_train_SLDS.shape[-1], # Input dimension
    #                 K=cfg.model.r2_dim, # number of sets of dynamics dimensions
    #                 D=cfg.model.r_dim, # latent dim
    #                 emissions="gaussian")

    # # Fit the model using Laplace-EM with a structured variational posterior
    # q_lem_elbos, q_lem = slds.fit(inputs_train_SLDS, method="laplace_em",
    #                             variational_posterior="structured_meanfield",
    #                             initialize=False,
    #                             num_iters=100, alpha=0.0,)


    # posterior = slds._make_variational_posterior( variational_posterior="structured_meanfield",datas=inputs_test_SLDS,inputs=None, masks=None, tags=None,method="laplace_em")
    # q_lem_x = posterior.mean_continuous_states[0]

    # # # Find the permutation that matches the true and inferred states
    # slds.permute(find_permutation(full_state_z, slds.most_likely_states(q_lem_x, inputs_test_SLDS)))
    # q_lem_z = slds.most_likely_states(q_lem_x, inputs_test_SLDS)

    # # Smooth the data under the variational posterior
    # q_lem_y = slds.smooth(q_lem_x, inputs_test_SLDS)

    # with open(cfg.paths.log_dir/f'ssm_slds_test_full_{cfg.model.r_dim}D_{cfg.model.r2_dim}K_{cfg.dataset.ssm_params.seed}seed.pickle', 'wb') as handle:
    #     pickle.dump(slds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(posterior, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(q_lem_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(q_lem_z, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(q_lem_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # ###### rSLDS
    # print('Fitting rSLDS with Laplace-EM')
    # rslds = ssm.SLDS(N=inputs_train_SLDS.shape[-1], # Input dimension
    #                 K=cfg.model.r2_dim, # number of sets of dynamics dimensions
    #                 D=cfg.model.r_dim, # latent dim
    #                 transitions="recurrent",
    #                 emissions="gaussian",
    #                 single_subspace=True)
    
    # # rslds.initialize(inputs_train_SLDS)
    # q_elbos_lem, q_lem = rslds.fit(inputs_train_SLDS, method="laplace_em",
    #                             variational_posterior="structured_meanfield",
    #                             initialize=False, num_iters=100, alpha=0.0)
    # # xhat_lem = q_lem.mean_continuous_states[0]
    # # rslds.permute(find_permutation(full_state_z, rslds.most_likely_states(xhat_lem, inputs_test_SLDS)))
    # # zhat_lem = rslds.most_likely_states(xhat_lem, inputs_test_SLDS)

    # posterior = rslds._make_variational_posterior( variational_posterior="structured_meanfield",datas=inputs_test_SLDS,inputs=None, masks=None, tags=None,method="laplace_em")
    # q_lem_x = posterior.mean_continuous_states[0]

    # # # Find the permutation that matches the true and inferred states
    # rslds.permute(find_permutation(full_state_z, rslds.most_likely_states(q_lem_x, inputs_test_SLDS)))
    # q_lem_z = rslds.most_likely_states(q_lem_x, inputs_test_SLDS)

    # # Smooth the data under the variational posterior
    # q_lem_y = rslds.smooth(q_lem_x, inputs_test_SLDS)
    
    # with open(cfg.paths.log_dir/f'ssm_rslds_test_full_{cfg.model.r_dim}D_{cfg.model.r2_dim}K_{cfg.dataset.ssm_params.seed}seed.pickle', 'wb') as handle:
    #     pickle.dump(rslds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(posterior, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(q_lem_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(q_lem_z, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(q_lem_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parse_hydra_config()
    