import numpy as np
import jax.numpy as jnp
import jax.random
from functools import partial
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
    if (cfg.dataset.name == 'SLDS') or (cfg.dataset.name == 'SSM'):
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
        # === Feature extraction based on feature_type ===
    feature_type = cfg.train.get('feature_type', 'raw')

    if feature_type == 'option_a':
        # Option A: Positions + Velocities + Social features
        process_inputs = partial(calc_option_a_features, normalize=cfg.train.normalize)
    elif feature_type == 'option_b':
        # Option B: Egocentric features
        process_inputs = partial(calc_option_b_features, normalize=cfg.train.normalize)
    elif feature_type == 'raw':
        # Use raw positions (28 features)
        process_inputs = partial(calc_raw_features, normalize=cfg.train.normalize)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}. Choose from 'raw', 'option_a', 'option_b', or 'basic'.")

    data_path = sorted(list(cfg.paths.data_dir.glob('*train.npy')))[0]
    train_data_dict = np.load(data_path,allow_pickle=True).item()
    sequence_ids = list(train_data_dict['annotator-id_0'].keys())
    data_path = sorted(list(cfg.paths.data_dir.glob('*test.npy')))[0]
    test_data_dict = np.load(data_path,allow_pickle=True).item()
    vocabulary = train_data_dict['annotator-id_0'][sequence_ids[0]]['metadata']['vocab']
    keypoint_names = ['nose', 'ear_left', 'ear_right', 'neck', 'hip_left', 'hip_right', 'tail_base']
    pose_estimates_train = [process_inputs(data=train_data_dict['annotator-id_0'][j]['keypoints']) for j in list(train_data_dict['annotator-id_0'].keys())]
    pose_estimates_test = [process_inputs(data=test_data_dict['annotator-id_0'][j]['keypoints']) for j in list(test_data_dict['annotator-id_0'].keys())]
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

    return train_inputs,test_inputs,annotations_train,annotations_test,vocabulary,keypoint_names, pos_shape_train, pos_shape_test

def calc_raw_features(data, normalize=False):
    """
    Raw Positions: 28 features

    Extracts raw x,y positions for all 14 keypoints (2 mice × 7 keypoints × 2 coords).

    Args:
        data: numpy array of shape (T, 28) where 28 = 2 mice × 7 keypoints × 2 coords
              Keypoint order: nose, ear_left, ear_right, neck, hip_left, hip_right, tail_base
              
    Returns:
        numpy array of shape (T, 28): raw position feature vector
    """
    T = data.shape[0]
    data = data.transpose(0,1,3,2)

    # Reshape data: (T, 14, 2)
    reshaped = data.reshape(T, 14, 2)
    if normalize:
        reshaped = reshaped / np.max(np.abs(reshaped))
        
    # Flatten to (T, 28)
    features = reshaped.reshape(T, -1)

    return features

def calc_option_a_features(data, normalize=False):
    """
    Option A: Positions + Velocities + Social Features

    Extracts comprehensive features including:
    - Raw x,y positions for all 14 keypoints (28 features)
    - Velocities (dx/dt, dy/dt) for all 14 keypoints (28 features)
    - Social interaction features (9 features):
        * Neck-to-neck distance (1)
        * Nose-to-nose distance (1)
        * Relative angle between mice: cos, sin (2)
        * Body orientation mouse 1: cos, sin (2)
        * Body orientation mouse 2: cos, sin (2)
        * Approach velocity (1)

    Args:
        data: numpy array of shape (T, 28) where 28 = 2 mice × 7 keypoints × 2 coords
              Keypoint order: nose, ear_left, ear_right, neck, hip_left, hip_right, tail_base
              Layout: [m1_x0, m1_y0, ..., m1_x6, m1_y6, m2_x0, m2_y0, ..., m2_x6, m2_y6]

    Returns:
        numpy array of shape (T, 65): concatenated feature vector
    """
    T = data.shape[0]
    data = data.transpose(0,1,3,2)

    # Reshape data: (T, 2, 7, 2) where dims are [time, mouse, keypoint, xy]
    # First reshape to (T, 14, 2) then to (T, 2, 7, 2)
    reshaped = data.reshape(T, 14, 2)
    m1 = data[:, 0, :]   # Mouse 1: (T, 7, 2)
    m2 = data[:, 1, :]   # Mouse 2: (T, 7, 2)

    # === 1. Raw positions (28 features) ===
    positions = reshaped  # Keep original layout
    if normalize:
        # social_features = social_features / np.max(np.abs(social_features),axis=-1,keepdims=True)
        positions = positions / np.max(np.abs(positions))
        m1 = m1 / np.max(np.abs(reshaped),axis=(0,1))
        m2 = m2 / np.max(np.abs(reshaped),axis=(0,1))

    # === 2. Velocities (28 features) ===
    # Use forward differences for all timesteps, padding last timestep
    velocities = np.zeros_like(positions)
    velocities[:-1] = np.diff(positions, axis=0)
    velocities[-1] = velocities[-2]  # Repeat last velocity

    # === 3. Social Features (8 features) ===
    social_features = []

    # 3a. Neck-to-neck distance (keypoint index 3)
    neck_dist = np.linalg.norm(m1[:, 3, :] - m2[:, 3, :], axis=1)
    social_features.append(neck_dist)

    # 3b. Nose-to-nose distance (keypoint index 0)
    nose_dist = np.linalg.norm(m1[:, 0, :] - m2[:, 0, :], axis=1)
    social_features.append(nose_dist)

    # 3c. Relative angle between mice (based on neck positions)
    rel_angle = np.arctan2(m1[:, 3, 1] - m2[:, 3, 1], m1[:, 3, 0] - m2[:, 3, 0])
    social_features.append(np.cos(rel_angle))
    social_features.append(np.sin(rel_angle))

    # 3d. Body orientation for each mouse (neck to tail_base)
    # Mouse 1: tail_base (idx 6) - neck (idx 3)
    m1_body_angle = np.arctan2(m1[:, 6, 1] - m1[:, 3, 1], m1[:, 6, 0] - m1[:, 3, 0])
    social_features.append(np.cos(m1_body_angle))
    social_features.append(np.sin(m1_body_angle))

    # Mouse 2: tail_base (idx 6) - neck (idx 3)
    m2_body_angle = np.arctan2(m2[:, 6, 1] - m2[:, 3, 1], m2[:, 6, 0] - m2[:, 3, 0])
    social_features.append(np.cos(m2_body_angle))
    social_features.append(np.sin(m2_body_angle))

    # 3e. Approach velocity (derivative of neck distance)
    approach_vel = np.zeros(T)
    approach_vel[:-1] = np.diff(neck_dist)
    approach_vel[-1] = approach_vel[-2]  # Repeat last value
    social_features.append(approach_vel)

    # Stack all social features: (T, 8)
    social_features = np.stack(social_features, axis=1)

    # === Concatenate all features ===
    features = np.concatenate([positions.reshape(T, -1), velocities.reshape(T, -1), social_features], axis=-1)

    return features


def calc_option_b_features(data, normalize=False):
    """
    Option B: Egocentric (Body-Centric) Features

    Extracts rotation and translation invariant features:
    - Per-mouse egocentric keypoint positions relative to neck (24 features: 2 mice × 6 keypoints × 2 coords)
    - Per-mouse egocentric velocities (24 features)
    - Per-mouse body pose features (8 features: 2 mice × 4 pose metrics)
    - Social interaction features in relative coordinates (8 features)

    Args:
        data: numpy array of shape (T, 28) where 28 = 2 mice × 7 keypoints × 2 coords
              Keypoint order: nose, ear_left, ear_right, neck, hip_left, hip_right, tail_base

    Returns:
        numpy array of shape (T, 64): concatenated egocentric feature vector
    """
    T = data.shape[0]
    data = data.transpose(0,1,3,2)

    # Reshape data: (T, 14, 2)
    reshaped = data.reshape(T, 14, 2)
    m1 = data[:, 0, :]   # Mouse 1: (T, 7, 2)
    m2 = data[:, 1, :]   # Mouse 2: (T, 7, 2)
    if normalize:
        # social_features = social_features / np.max(np.abs(social_features),axis=-1,keepdims=True)
        m1 = m1 / np.max(np.abs(reshaped),axis=(0,1))
        m2 = m2 / np.max(np.abs(reshaped),axis=(0,1))
        
    # === 1. Per-mouse egocentric positions (24 features) ===
    # Use neck (index 3) as reference point for each mouse
    m1_neck = m1[:, 3:4, :]  # (T, 1, 2)
    m2_neck = m2[:, 3:4, :]  # (T, 1, 2)

    # Translate keypoints relative to neck (exclude neck itself)
    m1_ego_pos = []
    m2_ego_pos = []

    for i in range(7):
        if i != 3:  # Skip neck
            m1_ego_pos.append(m1[:, i, :] - m1_neck[:, 0, :])
            m2_ego_pos.append(m2[:, i, :] - m2_neck[:, 0, :])

    m1_ego_pos = np.stack(m1_ego_pos, axis=1).reshape(T, -1)  # (T, 12)
    m2_ego_pos = np.stack(m2_ego_pos, axis=1).reshape(T, -1)  # (T, 12)
    ego_positions = np.concatenate([m1_ego_pos, m2_ego_pos], axis=1)  # (T, 24)
    if normalize:
        ego_positions = ego_positions / np.max(np.abs(ego_positions))
        
    # === 2. Per-mouse egocentric velocities (24 features) ===
    ego_velocities = np.zeros_like(ego_positions)
    ego_velocities[:-1] = np.diff(ego_positions, axis=0)
    ego_velocities[-1] = ego_velocities[-2]  # Repeat last velocity

    # === 3. Per-mouse body pose features (8 features) ===
    pose_features = []

    # For each mouse:
    for mouse in [m1, m2]:
        # 3a. Body length (neck to tail_base distance)
        body_length = np.linalg.norm(mouse[:, 6, :] - mouse[:, 3, :], axis=1)
        pose_features.append(body_length)

        # 3b. Head width (distance between ears)
        head_width = np.linalg.norm(mouse[:, 1, :] - mouse[:, 2, :], axis=1)
        pose_features.append(head_width)

        # 3c. Body elongation ratio (front-back distance / left-right distance)
        # Front-back: nose to tail_base
        front_back = np.linalg.norm(mouse[:, 0, :] - mouse[:, 6, :], axis=1)
        # Left-right: hip_left to hip_right
        left_right = np.linalg.norm(mouse[:, 4, :] - mouse[:, 5, :], axis=1) + 1e-6  # Avoid division by zero
        elongation = front_back / left_right
        pose_features.append(elongation)

        # 3d. Nose-to-neck distance (head extension)
        nose_neck_dist = np.linalg.norm(mouse[:, 0, :] - mouse[:, 3, :], axis=1)
        pose_features.append(nose_neck_dist)

    pose_features = np.stack(pose_features, axis=1)  # (T, 8)

    # === 4. Social interaction features (8 features) ===
    social_features = []

    # 4a. Inter-mouse distance (neck-to-neck)
    neck_dist = np.linalg.norm(m1[:, 3, :] - m2[:, 3, :], axis=1)
    social_features.append(neck_dist)

    # 4b. Relative position of mouse 2 w.r.t. mouse 1 (distance and angle)
    rel_vec = m2[:, 3, :] - m1[:, 3, :]  # Vector from m1 neck to m2 neck
    rel_angle = np.arctan2(rel_vec[:, 1], rel_vec[:, 0])
    social_features.append(np.cos(rel_angle))
    social_features.append(np.sin(rel_angle))

    # 4c. Relative body orientations
    m1_body_angle = np.arctan2(m1[:, 6, 1] - m1[:, 3, 1], m1[:, 6, 0] - m1[:, 3, 0])
    m2_body_angle = np.arctan2(m2[:, 6, 1] - m2[:, 3, 1], m2[:, 6, 0] - m2[:, 3, 0])

    # Relative orientation (angle difference)
    rel_orientation = m1_body_angle - m2_body_angle
    social_features.append(np.cos(rel_orientation))
    social_features.append(np.sin(rel_orientation))

    # 4d. Approach velocity
    approach_vel = np.zeros(T)
    approach_vel[:-1] = np.diff(neck_dist)
    approach_vel[-1] = approach_vel[-2]
    social_features.append(approach_vel)

    # 4e. Nose-to-nose distance
    nose_dist = np.linalg.norm(m1[:, 0, :] - m2[:, 0, :], axis=1)
    social_features.append(nose_dist)

    # 4f. Alignment score (how aligned are the body axes)
    alignment = np.cos(m1_body_angle - m2_body_angle)
    social_features.append(alignment)

    social_features = np.stack(social_features, axis=1)  # (T, 8)
    # if normalize:
        # social_features = social_features / np.max(np.abs(social_features),axis=-1,keepdims=True)
        
    # === Concatenate all features ===
    features = np.concatenate([ego_positions.reshape(T, -1), ego_velocities.reshape(T, -1), pose_features.reshape(T, -1), social_features], axis=-1)

    return features
