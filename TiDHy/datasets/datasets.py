import ssm
import numpy as np
import pandas as pd
import numpy.random as npr
from TiDHy.utils import random_rotation, set_seed
import TiDHy.utils.io_dict_to_hdf5 as ioh5
import torch

def load_dataset(cfg):
    """ Load dataset based on the configuration file.
    To add datasets to the code, add a new elif statement with the dataset name and the corresponding function to load the dataset.

    Args:
        cfg (OmegaConf): configuration file with dataset parameters.

    Returns:
        data_dict: dictionary with the dataset that should include: inputs_train, inputs_val, inputs_test.
    """
    print('Creating Dataset: {}'.format(cfg.dataset.name))
    data_dict = {}
    if cfg.dataset.name == 'CalMS21':
        inputs_train,inputs_test,annotations_train,annotations_test,vocabulary,keypoint_names = load_CalMS21_dataset(cfg)
        if cfg.delay.delay_embed:
            cfg.delay['orig_input_size'] = inputs_train.shape[-1]
            inputs_train = delay_embedding(inputs_train,cfg.delay.delay_tau,cfg.delay.skipt)
            inputs_test = delay_embedding(inputs_test,cfg.delay.delay_tau,cfg.delay.skipt)
        data_dict['inputs_train'] = inputs_train
        data_dict['inputs_val'] = inputs_test
        data_dict['inputs_test'] = inputs_test
        data_dict['annotations_train'] = annotations_train
        data_dict['annotations_test'] = annotations_test
        data_dict['annotations_val'] = annotations_test
        data_dict['vocabulary'] = vocabulary
        data_dict['keypoint_names'] = keypoint_names
    elif cfg.dataset.name == 'LDS':
        ssm_params = cfg.dataset.ssm_params
        lds_dict, data_dict = partial_superposition_LDS(cfg,ssm_params,**ssm_params)
        data_dict['states_x'] = np.transpose(data_dict['states_x'],(1,0,2)).reshape(ssm_params['time_bins_train'],-1)
        data_dict['states_x_test'] = np.transpose(data_dict['states_x_test'],(1,0,2)).reshape(ssm_params['time_bins_test'],-1)
        data_dict['states_x_val'] = np.transpose(data_dict['states_x_val'],(1,0,2)).reshape(ssm_params['time_bins_test'],-1)
        if cfg.delay.delay_embed:
            cfg.model.orig_input_size = inputs_train.shape[-1]
            inputs_train = delay_embedding(inputs_train,cfg.delay.delay_tau)
            inputs_test = delay_embedding(inputs_test,cfg.delay.delay_tau)
            if cfg.delay.delay_pcs:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=cfg.delay.delay_tau)
                pca.fit(inputs_train)
                ncomps = np.argmin(np.abs(np.cumsum(pca.explained_variance_ratio_)-.95))
                inputs_train = inputs_train[:,:ncomps]
                inputs_test = inputs_test[:,:ncomps]
                cfg.delay.orig_delay_tau = cfg.delay.delay_tau
                cfg.delay.delay_tau = ncomps
    elif (cfg.dataset.name == 'SLDS') | (cfg.dataset.name == 'SSM'):
        ssm_params = cfg.dataset.ssm_params
        lds_dict, data_dict = partial_superposition_SLDS(cfg,ssm_params,**ssm_params)
        data_dict['states_x'] = np.transpose(data_dict['states_x'],(1,0,2)).reshape(ssm_params['time_bins_train'],-1)
        data_dict['states_x_test'] = np.transpose(data_dict['states_x_test'],(1,0,2)).reshape(ssm_params['time_bins_test'],-1)
        data_dict['states_x_val'] = np.transpose(data_dict['states_x_val'],(1,0,2)).reshape(ssm_params['time_bins_test'],-1)
    elif cfg.dataset.name == 'AnymalTerrain':
        data_dict = load_isaacgym_dataset(cfg)

    assert 'inputs_train' in data_dict.keys(), 'inputs_train not in data_dict'
    assert 'inputs_val' in data_dict.keys(), 'inputs_val not in data_dict'
    assert 'inputs_test' in data_dict.keys(), 'inputs_test not in data_dict'
    return data_dict, cfg


def partial_superposition_em(emissions,ssm_params):
    '''
    Take the emissions from multiple LDS systems and partially superimpose them by the overlap parameter.
    For example 2 LDS systems with 3 observed dimensions with overlap=1 would have 5 observed dimensions
    in the output where the last dimentions is the sum of the last two dimensions of the original emissions.
    emissions: (time_bins,Nlds,obs_dim)
    ssm_params: dict
    '''
    time_bins = emissions.shape[0]
    overlap_em = np.sum(emissions[:,:,-ssm_params['overlap']:],axis=1,keepdims=True).reshape(time_bins,-1)
    em = np.zeros((time_bins,ssm_params['obs_dim']*ssm_params['Nlds'] - (ssm_params['Nlds']-1)*ssm_params['overlap']))
    em[:,:-ssm_params['overlap']] = emissions[:,:,:-ssm_params['overlap']].reshape(time_bins,-1)
    em[:,-ssm_params['overlap']:] = overlap_em
    return em

def delay_embedding(signal,delay,skipt=1):
    '''
    Create a delay embedding of the signal.
    signal: (time_bins,obs_dim)
    delay: int
    '''
    delayed_sig = np.stack([np.hstack((signal[n:,m],np.zeros(n))) for m in range(signal.shape[1]) for n in range(0,skipt*delay,skipt)],axis=1)
    return delayed_sig
    
    
def load_isaacgym_dataset(cfg,r_thresh=100,train_size=200,test_size=100):
    data = np.load(cfg.paths.data_dir / 'robot_dataset.npy', allow_pickle=True).item()
    from sklearn.model_selection import train_test_split

    robot_type_0 = np.where(data['robot_type'] == 0)[0]
    robot_type_1 = np.where(data['robot_type'] == 1)[0]
    train_size = cfg.dataset.train.train_size
    test_size  = cfg.dataset.train.test_size
    if cfg.dataset.train.single_ani: 
        train_idx1, test_idx1, train_idx2, test_idx2 = train_test_split(robot_type_0[:2000],robot_type_1[:2000], train_size=train_size, random_state=42)
        train_idx = train_idx1
        val_idx   = test_idx1[:test_size]
        test_idx  = test_idx2[:test_size]
    else:
        train_idx1, test_idx1, train_idx2, test_idx2 = train_test_split(robot_type_0[:2000],robot_type_1[:2000], train_size=train_size, random_state=42)
        train_idx  = np.concatenate([train_idx1,train_idx2])
        val_idx  = np.concatenate([test_idx1[test_size//2:test_size],test_idx2[test_size//2:test_size]])
        test_idx = np.concatenate([test_idx1[:test_size//2],test_idx2[:test_size//2]])
    data_dict={}
    data_dict['train_idx'] = train_idx
    data_dict['val_idx'] = val_idx
    data_dict['test_idx'] = test_idx
    inputs_train = np.concatenate((np.swapaxes(data['dof_pos'][train_idx],1,2),np.swapaxes(data['dof_vel'][train_idx],1,2)),axis=-1)
    inputs_val = np.concatenate((np.swapaxes(data['dof_pos'][val_idx],1,2),np.swapaxes(data['dof_vel'][val_idx],1,2)),axis=-1)
    inputs_test = np.concatenate((np.swapaxes(data['dof_pos'][test_idx],1,2),np.swapaxes(data['dof_vel'][test_idx],1,2)),axis=-1)
    if cfg.train.normalize_obs:
        inputs_train = (inputs_train/np.max(np.abs(inputs_train),axis=(-2),keepdims=True))
        inputs_val = (inputs_val/np.max(np.abs(inputs_val),axis=(-2),keepdims=True))
        inputs_test = (inputs_test/np.max(np.abs(inputs_test),axis=(-2),keepdims=True))
    terrain_type_train = data['terrain_type'][train_idx]
    terrain_type_val = data['terrain_type'][val_idx]
    terrain_type_test = data['terrain_type'][test_idx]
    robot_type_train = data['robot_type'][train_idx]
    robot_type_val = data['robot_type'][val_idx]
    robot_type_test = data['robot_type'][test_idx]
    
    terrain_difficulty_train = data['terrain_difficulty'][train_idx]
    terrain_difficulty_val = data['terrain_difficulty'][val_idx]
    terrain_difficulty_test = data['terrain_difficulty'][test_idx]
    terrain_slope_train = data['terrain_slope'][train_idx]
    terrain_slope_val = data['terrain_slope'][val_idx]
    terrain_slope_test = data['terrain_slope'][test_idx]
    command_vel_train = data['command_vel'][train_idx]
    command_vel_val = data['command_vel'][val_idx]
    command_vel_test = data['command_vel'][test_idx]
    
    
    # ##### Train/Val Dataset #####
    if inputs_train.shape[1]%cfg.train.sequence_length !=0:
        data_dict['inputs_train']   = inputs_train[:,:-(inputs_train.shape[1]%cfg.train.sequence_length)]
        data_dict['inputs_val']     = inputs_val[:,:-(inputs_val.shape[1]%cfg.train.sequence_length)]
        data_dict['inputs_test']    = inputs_test[:,:-(inputs_test.shape[1]%cfg.train.sequence_length)]
        data_dict['terrain_train']  = terrain_type_train[train_idx,:-(inputs_train.shape[1]%cfg.train.sequence_length)].squeeze()
        data_dict['terrain_val']    = terrain_type_val[val_idx,:-(inputs_val.shape[1]%cfg.train.sequence_length)].squeeze()
        data_dict['terrain_test']   = terrain_type_test[test_idx,:-(inputs_test.shape[1]%cfg.train.sequence_length)].squeeze()
        data_dict['terrain_difficulty_train'] = terrain_difficulty_train[train_idx,:-(inputs_train.shape[1]%cfg.train.sequence_length)].squeeze()
        data_dict['terrain_difficulty_val'] = terrain_difficulty_val[val_idx,:-(inputs_val.shape[1]%cfg.train.sequence_length)].squeeze()
        data_dict['terrain_difficulty_test'] = terrain_difficulty_test[test_idx,:-(inputs_test.shape[1]%cfg.train.sequence_length)].squeeze()
        data_dict['terrain_slope_train'] = terrain_slope_train[train_idx,:-(inputs_train.shape[1]%cfg.train.sequence_length)].squeeze()
        data_dict['terrain_slope_val'] = terrain_slope_val[val_idx,:-(inputs_val.shape[1]%cfg.train.sequence_length)].squeeze()
        data_dict['terrain_slope_test'] = terrain_slope_test[test_idx,:-(inputs_test.shape[1]%cfg.train.sequence_length)].squeeze()
        data_dict['command_vel_train'] = command_vel_train[train_idx,:-(inputs_train.shape[1]%cfg.train.sequence_length)].squeeze()
        data_dict['command_vel_val'] = command_vel_val[val_idx,:-(inputs_val.shape[1]%cfg.train.sequence_length)].squeeze()
        data_dict['command_vel_test'] = command_vel_test[test_idx,:-(inputs_test.shape[1]%cfg.train.sequence_length)].squeeze()
    else:
        data_dict['inputs_train'] = inputs_train
        data_dict['inputs_val'] = inputs_val
        data_dict['inputs_test'] = inputs_test
        data_dict['terrain_train']  = terrain_type_train.squeeze()
        data_dict['terrain_val']    = terrain_type_val.squeeze()
        data_dict['terrain_test']   = terrain_type_test.squeeze()
        data_dict['terrain_difficulty_train'] = terrain_difficulty_train.squeeze()
        data_dict['terrain_difficulty_val'] = terrain_difficulty_val.squeeze()
        data_dict['terrain_difficulty_test'] = terrain_difficulty_test.squeeze()
        data_dict['terrain_slope_train'] = terrain_slope_train.squeeze()
        data_dict['terrain_slope_val'] = terrain_slope_val.squeeze()
        data_dict['terrain_slope_test'] = terrain_slope_test.squeeze()
        data_dict['command_vel_train'] = command_vel_train.squeeze()
        data_dict['command_vel_val'] = command_vel_val.squeeze()
        data_dict['command_vel_test'] = command_vel_test.squeeze()
        
    data_dict['robot_type_train'] = robot_type_train
    data_dict['robot_type_val'] = robot_type_val
    data_dict['robot_type_test'] = robot_type_test
    
    return data_dict


def load_CalMS21_dataset(cfg):
    '''
    Load the CalMS21 dataset.
    cfg.paths.data_dir: Path
    params: dict
    '''
    data_path = sorted(list(cfg.paths.data_dir.glob('*train.npy')))[0]
    train_data_dict = np.load(data_path,allow_pickle=True).item()
    sequence_ids = list(train_data_dict['annotator-id_0'].keys())
    data_path = sorted(list(cfg.paths.data_dir.glob('*test.npy')))[0]
    test_data_dict = np.load(data_path,allow_pickle=True).item()
    vocabulary = train_data_dict['annotator-id_0'][sequence_ids[0]]['metadata']['vocab']
    keypoint_names = ['nose', 'ear_left', 'ear_right', 'neck', 'hip_left', 'hip_right', 'tail_base']
    pose_estimates_train = [train_data_dict['annotator-id_0'][j]['keypoints'] for j in list(train_data_dict['annotator-id_0'].keys())]
    pose_estimates_test = [test_data_dict['annotator-id_0'][j]['keypoints'] for j in list(test_data_dict['annotator-id_0'].keys())]
    annotations_train = [train_data_dict['annotator-id_0'][j]['annotations'] for j in list(train_data_dict['annotator-id_0'].keys())]
    annotations_test = [test_data_dict['annotator-id_0'][j]['annotations'] for j in list(test_data_dict['annotator-id_0'].keys())]
    train_data = np.concatenate([seq.reshape(seq.shape[0],-1) for seq in pose_estimates_train if seq.shape[0] > cfg.train['sequence_length']],axis=0)
    test_data = np.concatenate([seq.reshape(seq.shape[0],-1) for seq in pose_estimates_test if seq.shape[0] > cfg.train['sequence_length']],axis=0)
    annotations_train = np.concatenate(annotations_train,axis=0)
    annotations_test = np.concatenate(annotations_test,axis=0)
    train_inputs = train_data[:-(train_data.shape[0]%cfg.train.sequence_length)]
    test_inputs = test_data[:-(test_data.shape[0]%cfg.train.sequence_length)]
    annotations_train = annotations_train[:-(annotations_train.shape[0]%cfg.train.sequence_length)]
    annotations_test = annotations_test[:-(annotations_test.shape[0]%cfg.train.sequence_length)]
    if cfg.train.normalize:
        train_inputs = train_inputs/np.max(np.abs(train_inputs))
        test_inputs = test_inputs/np.max(np.abs(test_inputs))
    if cfg.dataset.train.add_basic_features:
        if cfg.dataset.train.only_basic_features:
            train_inputs = calc_basic_features(train_inputs,only_basic=None)
            test_inputs = calc_basic_features(test_inputs,only_basic=None)
        else:
            train_inputs = calc_basic_features(train_inputs,only_basic=False)
            test_inputs = calc_basic_features(test_inputs,only_basic=False)
    # else:
    
    return train_inputs,test_inputs,annotations_train,annotations_test,vocabulary,keypoint_names

def calc_basic_features(data,only_basic=None):
    m1 = data[:,:14].reshape(-1,2,7)
    m2 = data[:,14:].reshape(-1,2,7)
    dist = np.linalg.norm(m1[:,:,3] - m2[:,:,3],ord=2,axis=1)
    angle = np.arctan2((m1[:,1,3] - m2[:,1,3]),(m1[:,0,3] - m2[:,0,3]))*180/np.pi
    cos_ang = np.cos(angle)
    sin_ang = np.sin(angle)
    if only_basic is None:
        data = np.stack([dist,cos_ang,sin_ang],axis=1)
    else:
        data = np.concatenate([data,np.stack([dist,cos_ang,sin_ang],axis=1)],axis=1)
    return data

def create_dyn_mat(desired_eigenvalues):
    from scipy import signal
    # Create an empty A matrix with the same dimensions as desired_eigenvalues
    A = np.zeros((len(desired_eigenvalues), len(desired_eigenvalues)))
    # Create the system with the desired eigenvalues using pole placement
    B = np.random.randn(len(desired_eigenvalues),len(desired_eigenvalues))  
    # Calculate the state-space representation
    poles = signal.place_poles(A, B,desired_eigenvalues)
    A0 = A-B@poles.gain_matrix
    return A0

def partial_superposition_LDS(cfg, ssm_params, timescales=None, normalize=True, random_projection=True, partial_sup=True, partial_obs=False, full_sup=False, seed=0, saved_evals=True,**kwargs):
    """Generate data from a linear dynamical system (LDS) with partial superposition."""
    ##### Fast to slow #####
    timescales = [-.025,-.25,-.1,-.5,-.015,-.75]
    freq = np.zeros(len(timescales)).astype(complex)
    freq.imag = [.0075,.05,.025,.075,.01,.075]
    set_eigvalues = np.array([[np.exp(timescales) + 2*np.pi*np.array(freq)],[np.exp(timescales) - 2*np.pi*np.array(freq)]]).squeeze().T

    # Set the random seed
    filename = 'LDS_N{}_LD{}_OD{}_seed{}.h5'.format(ssm_params['Nlds'],ssm_params['latent_dim'],ssm_params['obs_dim'],ssm_params['seed'])
    if (cfg.paths.data_dir/filename).exists():
        data_dict = ioh5.load(cfg.paths.data_dir / filename)
        inputs_train = data_dict['inputs_train']
        inputs_val = data_dict['inputs_val']
        inputs_test = data_dict['inputs_test']
        states_x = data_dict['states_x']
        states_x_val = data_dict['states_x_val']
        states_x_test = data_dict['states_x_test']
        As = data_dict['As']
        bs = data_dict['bs']
        lds_dict={}
    else:  
        npr.seed(seed)
        torch.manual_seed(seed)
        # Make an LDS with somewhat interesting dynamics parameters
        lds_dict = {}
        states_x, states_x_test, states_x_val = [], [], []
        inputs_train, inputs_test, inputs_val = [], [], []
        As,bs = [],[]
        if ~isinstance(timescales, np.ndarray):
            timescales=np.array(timescales)
        p = 0
        for i in range(ssm_params['Nlds']):
            true_lds = ssm.LDS(ssm_params['obs_dim'],ssm_params['latent_dim'], emissions="gaussian")
            As_lds, bs_lds = [], []
            if saved_evals: 
                A0 = create_dyn_mat(set_eigvalues[i])
            else:
                # for n in range(ssm_params['n_disc_states']):
                neg = True if i % 2 == 0 else False
                theta = np.pi/2 * npr.rand()/(timescales[i])
                A0 = .95 * random_rotation(ssm_params['latent_dim'], theta=theta, neg=neg)
            b = np.zeros(ssm_params['latent_dim']) #npr.randn(ssm_params['latent_dim'])
            As_lds.append(A0)
            bs_lds.append(b)
            # Set the dynamics matrix (A) and the 
            true_lds.dynamics.As = np.stack(As_lds)
            true_lds.dynamics.bs = np.stack(bs_lds)
            x, y = true_lds.sample(ssm_params['time_bins_train'])
            x2, y2 = true_lds.sample(ssm_params['time_bins_test'])
            x3, y3 = true_lds.sample(ssm_params['time_bins_test'])
            if normalize:
                x = x/np.max(np.abs(x),axis=0,keepdims=True)
                x2 = x2/np.max(np.abs(x2),axis=0,keepdims=True)
                x3 = x3/np.max(np.abs(x3),axis=0,keepdims=True)
            states_x.append(x)
            states_x_test.append(x2)
            inputs_train.append(y)
            inputs_test.append(y2)
            inputs_val.append(y3)
            states_x_val.append(x3)
            As.append(true_lds.dynamics.As)
            bs.append(true_lds.dynamics.bs)

            lds_dict['{}'.format(i)] = true_lds


        states_x = np.stack(states_x)
        states_x_test = np.stack(states_x_test)
        states_x_val = np.stack(states_x_val)
        inputs_train = np.stack(inputs_train,axis=1)
        inputs_test = np.stack(inputs_test,axis=1)
        inputs_val = np.stack(inputs_val,axis=1)
        data_dict={'states_x':states_x,'states_x_test':states_x_test, 'states_x_val':states_x_val,
                    'inputs_train':inputs_train,'inputs_test':inputs_test, 'inputs_val':inputs_val,
                    'As':As,'bs':bs,'timescales':timescales}
        ioh5.save(cfg.paths.data_dir/filename,data_dict)
    
    set_seed(42)
    if partial_sup:
        ##### Partial Superposition #####
        set_seed(42)
        assert ssm_params['overlap'] <= ssm_params['obs_dim'], 'Overlap must be less than or equal to the number of observations'
        inputs_train = inputs_train.reshape(ssm_params['time_bins_train'],-1)
        inputs_test = inputs_test.reshape(ssm_params['time_bins_test'],-1)
        inputs_val = inputs_val.reshape(ssm_params['time_bins_test'],-1)
        ##### Constructing the mixing matrix #####
        sup_inds = np.stack([np.arange(n,ssm_params['obs_dim']*ssm_params['Nlds'],ssm_params['obs_dim']) for n in range(ssm_params['obs_dim'])])
        MixMat = np.zeros((ssm_params['obs_dim']*ssm_params['Nlds'],ssm_params['obs_dim']*ssm_params['Nlds']))
        for n in range(sup_inds.shape[0]):
            if n < ssm_params['overlap']:
                MixMat[sup_inds[n],n] = 1
            else:
                MixMat[sup_inds[n],sup_inds[n]] = 1
        idx = np.argwhere(np.all(MixMat[..., :] == 0, axis=0))
        MixMat = np.delete(MixMat, idx, axis=1)
        inputs_train = inputs_train@MixMat
        inputs_test = inputs_test@MixMat
        inputs_val = inputs_val@MixMat
    elif partial_obs:
        inputs_train = inputs_train[:,:,:-1].reshape(inputs_train.shape[0],-1)
        inputs_test = inputs_test[:,:,:-1].reshape(inputs_test.shape[0],-1)
        inputs_val = inputs_val[:,:,:-1].reshape(inputs_val.shape[0],-1)
    elif full_sup:
        inputs_train = np.sum(inputs_train,axis=(-1,-2))[:,np.newaxis]
        inputs_test = np.sum(inputs_test,axis=(-1,-2))[:,np.newaxis]
        inputs_val = np.sum(inputs_val,axis=(-1,-2))[:,np.newaxis]
    else:
        inputs_train = inputs_train.reshape(inputs_train.shape[0],-1)
        inputs_test = inputs_test.reshape(inputs_test.shape[0],-1)
        inputs_val = inputs_val.reshape(inputs_val.shape[0],-1)
    if random_projection:
        inputs_train = inputs_train@np.random.randn(inputs_train.shape[-1],ssm_params['rand_dim'])
        inputs_test = inputs_test@np.random.randn(inputs_test.shape[-1],ssm_params['rand_dim'])
        inputs_val = inputs_val@np.random.randn(inputs_val.shape[-1],ssm_params['rand_dim'])
    if normalize:
        inputs_train = inputs_train/np.max(np.abs(inputs_train),axis=0,keepdims=True)
        inputs_test = inputs_test/np.max(np.abs(inputs_test),axis=0,keepdims=True)
        inputs_val = inputs_val/np.max(np.abs(inputs_val),axis=0,keepdims=True)

    data_dict={'states_x':states_x,'states_x_test':states_x_test, 'states_x_val':states_x_val,
                'inputs_train':inputs_train,'inputs_test':inputs_test, 'inputs_val':inputs_val,
                'As':As,'bs':bs,'timescales':data_dict['timescales']}
    print('x Timescales: {}'.format(data_dict['timescales']))
    return lds_dict,data_dict

def partial_superposition_SLDS(cfg, ssm_params, timescales=None, normalize=False, random_projection=False, partial_sup=False, partial_obs=False, full_sup=False, seed=0,saved_evals=True,**kwargs):
    '''
    Create a dataset of LDS systems with partially superimposed inputs.
    ssm_params: dict
    '''
    ##### Fast to slow #####
    timescales = [-.025,-.25,-.1,-.5,-.015,-.75]
    freq = np.zeros(len(timescales)).astype(complex)
    freq.imag = [.0075,.05,.025,.075,.01,.075]
    set_eigvalues = np.array([[np.exp(timescales) + 2*np.pi*np.array(freq)],[np.exp(timescales) - 2*np.pi*np.array(freq)]]).squeeze().T

    # Set the random seed
    filename = 'SLDS_N{}_zD{}_xD{}_yD{}_seed{}.h5'.format(ssm_params['Nlds'],ssm_params['n_disc_states'],ssm_params['latent_dim'],ssm_params['obs_dim'],ssm_params['seed'])
    if (cfg.paths.data_dir/filename).exists():
        data_dict = ioh5.load(cfg.paths.data_dir / filename)
        inputs_train     = data_dict['inputs_train'][:ssm_params['time_bins_train']]
        inputs_val       = data_dict['inputs_val'][:ssm_params['time_bins_test']]
        inputs_test      = data_dict['inputs_test'][:ssm_params['time_bins_test']]
        states_z         = data_dict['states_z'][:ssm_params['time_bins_train']]
        states_z_test    = data_dict['states_z_test'][:ssm_params['time_bins_test']]
        states_z_val     = data_dict['states_z_val'][:ssm_params['time_bins_test']]
        states_x         = data_dict['states_x'][:ssm_params['time_bins_train']]
        states_x_val     = data_dict['states_x_val'][:ssm_params['time_bins_test']]
        states_x_test    = data_dict['states_x_test'][:ssm_params['time_bins_test']]
        As = data_dict['As']
        bs = data_dict['bs']
        lds_dict={}
    else:
        npr.seed(seed)
        torch.manual_seed(seed)
        # Make an LDS with somewhat interesting dynamics parameters
        lds_dict = {}
        states_z, states_z_test, states_z_val =  [], [], []
        states_x, states_x_test, states_x_val = [], [], []
        inputs_train, inputs_test, inputs_val = [], [], []
        As,bs = {},{}
        p = 0
        for i in range(ssm_params['Nlds']):
            true_lds = ssm.SLDS(ssm_params['obs_dim'], ssm_params['n_disc_states'],ssm_params['latent_dim'], emissions="gaussian")
            As_lds, bs_lds = [], []
            for n in range(ssm_params['n_disc_states']):
                if saved_evals:
                    A0 = create_dyn_mat(set_eigvalues[p])
                else:
                    a = 1 if i % 2 == 0 else -1
                    neg = True if i % 2 == 0 else False
                    theta = np.pi/2 * npr.rand()/(timescales[p])
                    A0 = .95 * random_rotation(ssm_params['latent_dim'], theta=theta, neg=neg)
                b = npr.uniform(low=-1,high=1,size=ssm_params['latent_dim']) #np.zeros(ssm_params['latent_dim']) #
                As_lds.append(A0)
                bs_lds.append(b)
                p+=1
            # Set the dynamics matrix (A) and the 
            true_lds.dynamics.As = np.stack(As_lds)
            true_lds.dynamics.bs = np.stack(bs_lds)
            K = ssm_params['n_disc_states']
            z_timescale = ssm_params['z_timescale'][i]
            Ps = z_timescale * np.eye(K) + (1-z_timescale) #* npr.rand(K, K)
            Ps /= Ps.sum(axis=1, keepdims=True)
            print(Ps)
            true_lds.transitions.log_Ps = np.log(Ps)
            p_states_x = 0
            z_count = 0
            while (np.min(p_states_x) < 0.45):
                z, x, y = true_lds.sample(ssm_params['time_bins_train'])
                _,counts = np.unique(z,return_counts=True)
                p_states_x = counts/np.sum(counts)
                z_count += 1
                print('Train',z_count, p_states_x)
            p_states_x = 0
            z_count = 0
            while (np.min(p_states_x) < 0.45):
                z2, x2, y2 = true_lds.sample(ssm_params['time_bins_test'])
                _,counts = np.unique(z2,return_counts=True)
                p_states_x = counts/np.sum(counts)
                z_count += 1
                print('Val',z_count, p_states_x)
            p_states_x = 0
            z_count = 0
            while (np.min(p_states_x) < 0.45):
                z3, x3, y3 = true_lds.sample(ssm_params['time_bins_test'])
                _,counts = np.unique(z3,return_counts=True)
                p_states_x = counts/np.sum(counts)
                z_count += 1
                print('Test',z_count, p_states_x)
            if normalize:
                x = x/np.max(np.abs(x),axis=0,keepdims=True)
                x2 = x2/np.max(np.abs(x2),axis=0,keepdims=True)
                x3 = x3/np.max(np.abs(x3),axis=0,keepdims=True)
            states_x.append(x)
            states_x_test.append(x2)
            states_z.append(z)
            states_z_test.append(z2)
            inputs_train.append(y)
            inputs_test.append(y2)
            inputs_val.append(y3)
            states_x_val.append(x3)
            states_z_val.append(z3)
            As['{}'.format(i)] = true_lds.dynamics.As
            bs['{}'.format(i)] = true_lds.dynamics.bs

            lds_dict['{}'.format(i)] = true_lds
        states_x = np.stack(states_x)
        states_x_test = np.stack(states_x_test)
        states_x_val = np.stack(states_x_val)
        states_z = np.stack(states_z)
        states_z_test = np.stack(states_z_test)
        states_z_val = np.stack(states_z_val)
        inputs_train = np.stack(inputs_train,axis=1)
        inputs_test = np.stack(inputs_test,axis=1)
        inputs_val = np.stack(inputs_val,axis=1)
        data_dict={'states_z':states_z,'states_z_test':states_z_test, 'states_z_val':states_z_val,
                   'states_x':states_x,'states_x_test':states_x_test, 'states_x_val':states_x_val,
                   'inputs_train':inputs_train,'inputs_test':inputs_test, 'inputs_val':inputs_val,
                   'As':As,'bs':bs,'timescales':timescales}
        ioh5.save(cfg.paths.data_dir/filename,data_dict)
        ##### Print Saved Data Info #####
        import itertools
        lst = list(itertools.product([0, 1], repeat=3))
        lst2 = list(itertools.product(['F', 'S'], repeat=3))
        full_state_z = np.zeros(ssm_params['time_bins_train'],dtype=int)
        # full_state_z = np.zeros(ssm_params['time_bins_train'],dtype=int)
        for n in range(len(lst)):
            full_state_z[np.apply_along_axis(lambda x: np.all(x == lst[n]),0,data_dict['states_z'])] = n
        print(np.unique(full_state_z,return_counts=True))
        full_state_z = np.zeros(ssm_params['time_bins_test'],dtype=int)
        # full_state_z = np.zeros(ssm_params['time_bins_train'],dtype=int)
        for n in range(len(lst)):
            full_state_z[np.apply_along_axis(lambda x: np.all(x == lst[n]),0,data_dict['states_z_test'])] = n
        print(np.unique(full_state_z,return_counts=True))
        _,zcounts=np.unique(full_state_z,return_counts=True)
        print("Test",zcounts/np.sum(zcounts))
        
    ##### Partial Superposition #####
    set_seed(seed)
    if partial_sup:
        set_seed(seed)
        ##### Partial Superposition #####
        assert ssm_params['overlap'] <= ssm_params['obs_dim'], 'Overlap must be less than or equal to the number of observations'
        inputs_train = inputs_train.reshape(ssm_params['time_bins_train'],-1)
        inputs_test = inputs_test.reshape(ssm_params['time_bins_test'],-1)
        inputs_val = inputs_val.reshape(ssm_params['time_bins_test'],-1)
        ##### Constructing the mixing matrix #####
        sup_inds = np.stack([np.arange(n,ssm_params['obs_dim']*ssm_params['Nlds'],ssm_params['obs_dim']) for n in range(ssm_params['obs_dim'])])
        MixMat = np.zeros((ssm_params['obs_dim']*ssm_params['Nlds'],ssm_params['obs_dim']*ssm_params['Nlds']))
        for n in range(sup_inds.shape[0]):
            if n < ssm_params['overlap']:
                MixMat[sup_inds[n],n] = 1
            else:
                MixMat[sup_inds[n],sup_inds[n]] = 1
        idx = np.argwhere(np.all(MixMat[..., :] == 0, axis=0))
        MixMat = np.delete(MixMat, idx, axis=1)
        inputs_train = inputs_train@MixMat
        inputs_test = inputs_test@MixMat
        inputs_val = inputs_val@MixMat
    elif partial_obs:
        inputs_train = inputs_train[:,:,:-1].reshape(inputs_train.shape[0],-1)
        inputs_test = inputs_test[:,:,:-1].reshape(inputs_test.shape[0],-1)
        inputs_val = inputs_val[:,:,:-1].reshape(inputs_val.shape[0],-1)
    elif full_sup:
        inputs_train = np.sum(inputs_train,axis=(-1,-2))[:,np.newaxis]
        inputs_test = np.sum(inputs_test,axis=(-1,-2))[:,np.newaxis]
        inputs_val = np.sum(inputs_val,axis=(-1,-2))[:,np.newaxis]
    else:
        inputs_train = inputs_train.reshape(inputs_train.shape[0],-1)
        inputs_test = inputs_test.reshape(inputs_test.shape[0],-1)
        inputs_val = inputs_val.reshape(inputs_val.shape[0],-1)
    if random_projection:
        set_seed(seed)
        RandMat = np.random.randn(inputs_train.shape[-1],ssm_params['rand_dim'])
        inputs_train = inputs_train@RandMat
        inputs_test = inputs_test@RandMat
        inputs_val = inputs_val@RandMat
    if normalize:
        inputs_train = inputs_train/np.max(np.abs(inputs_train),axis=0,keepdims=True)
        inputs_test = inputs_test/np.max(np.abs(inputs_test),axis=0,keepdims=True)
        inputs_val = inputs_val/np.max(np.abs(inputs_val),axis=0,keepdims=True)
    if cfg.delay.delay_embed: 
        inputs_train = delay_embedding(inputs_train,cfg.delay.delay_tau,skipt=cfg.delay.skipt)
        inputs_test = delay_embedding(inputs_test,cfg.delay.delay_tau,skipt=cfg.delay.skipt)
        inputs_val = delay_embedding(inputs_val,cfg.delay.delay_tau,skipt=cfg.delay.skipt)

    data_dict={'states_z':states_z,'states_z_test':states_z_test, 'states_z_val':states_z_val,
                'states_x':states_x,'states_x_test':states_x_test, 'states_x_val':states_x_val,
                'inputs_train':inputs_train,'inputs_test':inputs_test, 'inputs_val':inputs_val,
                'As':As,'bs':bs,'timescales':data_dict['timescales']}
    print('x Timescales: {}, z Timescales: {}'.format(data_dict['timescales'],ssm_params['z_timescale']))
    return lds_dict,data_dict


