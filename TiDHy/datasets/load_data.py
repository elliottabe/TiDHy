import numpy as np
import jax.numpy as jnp
import jax.random
import TiDHy.utils.io_dict_to_hdf5 as ioh5
from TiDHy.datasets.datasets_dynamax import partial_superposition_SLDS
from TiDHy.datasets.rossler_dataset import hierarchical_rossler_dataset

def load_data(cfg):
    
    """ Load dataset based on the configuration file.
    To add datasets to the code, add a new elif statement with the dataset name and the corresponding function to load the dataset.

    Args:
        cfg (OmegaConf): configuration file with dataset parameters.

    Returns:
        data_dict: dictionary with the dataset that should include: inputs_train, inputs_val, inputs_test.
    """
    if cfg.dataset.name == 'SLDS':
        ssm_params = cfg.dataset.ssm_params
        lds_dict, data_dict = partial_superposition_SLDS(cfg, ssm_params)
    elif cfg.dataset.name == 'AnymalTerrain':
        data_dict = load_isaacgym_dataset(cfg)
    elif cfg.dataset.name == 'CalMS21':
        data_dict = {}
        inputs_train,inputs_test,annotations_train,annotations_test,vocabulary,keypoint_names, pos_shape_train, pos_shape_test = load_CalMS21_dataset(cfg)
        data_dict['inputs_train'] = inputs_train
        data_dict['inputs_val'] = inputs_test
        data_dict['inputs_test'] = inputs_test
        data_dict['annotations_train'] = annotations_train
        data_dict['annotations_test'] = annotations_test
        data_dict['annotations_val'] = annotations_test
        data_dict['vocabulary'] = vocabulary
        data_dict['keypoint_names'] = keypoint_names
        data_dict['pos_shape_train'] = pos_shape_train
        data_dict['pos_shape_test'] = pos_shape_test
    elif cfg.dataset.name == 'Rossler':
        rossler_params = cfg.dataset.rossler_params
        data_dict = hierarchical_rossler_dataset(cfg, rossler_params)
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not recognized.")
    
    assert 'inputs_train' in data_dict.keys(), 'inputs_train not in data_dict'
    assert 'inputs_val' in data_dict.keys(), 'inputs_val not in data_dict'
    assert 'inputs_test' in data_dict.keys(), 'inputs_test not in data_dict'
    return data_dict



def stack_data(inputs, sequence_length, overlap=None):
    """
    Create overlapping windows from input sequences using vectorized operations.

    Args:
        inputs: Input array of shape (Time, Input_Size) or (Batch, Time, Input_Size)
        sequence_length: Length of each window
        overlap: Overlap between consecutive windows (default: sequence_length//2)

    Returns:
        Stacked windows of shape (num_windows, sequence_length, Input_Size)
        or (num_windows * Batch, sequence_length, Input_Size) for 3D input
    """
    if overlap is None:
        overlap = sequence_length // 2

    stride = overlap  # Step size between windows

    if len(inputs.shape) == 2:
        # (Time, Input_Size)
        time_len, input_size = inputs.shape

        # Calculate number of windows that fit
        num_windows = (time_len - sequence_length) // stride + 1
        if num_windows <= 0:
            return jnp.empty((0, sequence_length, input_size))

        # Pre-compute all start indices
        start_indices = jnp.arange(num_windows) * stride

        # Vectorized window extraction using advanced indexing
        # Create indices for all windows at once: (num_windows, sequence_length)
        window_indices = start_indices[:, None] + jnp.arange(sequence_length)[None, :]

        # Extract all windows at once: (num_windows, sequence_length, input_size)
        return inputs[window_indices]

    elif len(inputs.shape) == 3:
        # (Batch, Time, Input_Size)
        batch_size, time_len, input_size = inputs.shape

        # Calculate number of windows that fit
        num_windows = (time_len - sequence_length) // stride + 1
        if num_windows <= 0:
            return jnp.empty((0, sequence_length, input_size))

        # Pre-compute all start indices
        start_indices = jnp.arange(num_windows) * stride

        # Vectorized window extraction
        # Create indices: (num_windows, sequence_length)
        window_indices = start_indices[:, None] + jnp.arange(sequence_length)[None, :]

        # Extract all windows for all batches: (batch, num_windows, sequence_length, input_size)
        windowed = inputs[:, window_indices, :]

        # Reshape to (batch * num_windows, sequence_length, input_size)
        return windowed.reshape(-1, sequence_length, input_size)

    else:
        raise ValueError(f"Input must be 2D or 3D, got shape {inputs.shape}")
    
    
def load_isaacgym_dataset(cfg,r_thresh=100,train_size=200,test_size=100):
    data = np.load(cfg.paths.data_dir / 'robot_dataset.npy', allow_pickle=True).item()
    from sklearn.model_selection import train_test_split

    robot_type_0 = np.where(data['robot_type'] == 0)[0]
    robot_type_1 = np.where(data['robot_type'] == 1)[0]
    train_size = cfg.dataset.train_size
    test_size  = cfg.dataset.test_size
    if cfg.dataset.single_ani: 
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
    pos_shape_train = [train_data_dict['annotator-id_0'][j]['keypoints'].shape for j in list(train_data_dict['annotator-id_0'].keys())]
    pos_shape_test = [test_data_dict['annotator-id_0'][j]['keypoints'].shape for j in list(test_data_dict['annotator-id_0'].keys())]
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
    if cfg.train.add_basic_features:
        if cfg.train.only_basic_features:
            train_inputs = calc_basic_features(train_inputs,only_basic=None)
            test_inputs = calc_basic_features(test_inputs,only_basic=None)
        else:
            train_inputs = calc_basic_features(train_inputs,only_basic=False)
            test_inputs = calc_basic_features(test_inputs,only_basic=False)
    # else:
    
    return train_inputs,test_inputs,annotations_train,annotations_test,vocabulary,keypoint_names, pos_shape_train, pos_shape_test

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
