import os
import json
import yaml
import logging
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from omegaconf import OmegaConf
from tqdm.auto import tqdm
from pathlib import Path


import matplotlib as mpl
import TiDHy.utils.io_dict_to_hdf5 as ioh5

##### Custom Resolver #####
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if_multi', lambda pred, a, b: a if pred.name=='MULTIRUN' else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)


def requires_grad(parameters, flag=True):
    """Batch set requires_grad flag for all parameters in a list of parameters.

    Args:
        parameters (list): A list of torch.nn.Parameter objects.
        flag (bool, optional): Requires grad flag. Defaults to True.
    """
    for p in parameters:
        p.requires_grad = flag

def set_logger(cfg,log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        if 'checkpoint' in cfg and cfg['checkpoint'] is not None and cfg['checkpoint'] !='':
            file_handler = logging.FileHandler(log_path, 'a+')
        else:
            file_handler = logging.FileHandler(log_path, 'w+')
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

        # logger.addHandler(TqdmLoggingHandler)
    # return logger

def save_dict_to_file(d, config_path):
    """Saves dict of floats in json or yaml file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        config_path: (string) path to json file
    """
    if config_path.endswith('.yaml'):    
        with open(config_path, 'w') as f:
            yaml.dump(d, f)
    elif config_path.endswith('.json'):
        with open(config_path, 'w') as f:
            # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
            d = {k: float(v) for k, v in d.items()}
            json.dump(d, f, indent=4)

def load_cfg(cfg_path, default_model_config):
    """ Load configuration file and merge with default model configuration

    Args:
        cfg_path (string): path to configuration file
        default_model_config (string): path to default model configuration file

    Returns:
        cfg: returns the merged configuration data
    """
    params = OmegaConf.load(default_model_config)
    cfg = OmegaConf.load(cfg_path)
    for k in cfg.paths.keys():
        if k != 'user':
            cfg.paths[k] = Path(cfg.paths[k])
            cfg.paths[k].mkdir(parents=True, exist_ok=True)
    params_curr = cfg.dataset.model
    cfg.dataset.model = OmegaConf.merge(params, params_curr)
    return cfg

def save_checkpoint(state, is_best, checkpoint, epoch, filename=None):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        filename: (string) non-default name for checkpoint file. If None (default), filename is `last.pth.tar` 
    """
    filename = 'last_{}.pth.tar'.format(epoch) if filename is None else filename
    filepath = os.path.join(checkpoint, filename)
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def load_results(model, dataloader, data_dict, device, cfg, epoch, seq_len=None,rerecord=False):
    if seq_len is None:
        seq_len = cfg.train.sequence_length
    if (cfg.paths.log_dir / 'results.h5').exists() & (rerecord==False):
        result_dict = ioh5.load(cfg.paths.log_dir / 'results.h5')
        for key in result_dict.keys():
            if isinstance(result_dict[key],dict):
                result_dict[key] = [result_dict[key][key2] for key2 in result_dict[key].keys()]
    elif ((cfg.paths.log_dir/'temp_results_{}_T{:04d}.h5'.format(epoch,seq_len)).exists()==False) | (rerecord==True):
        set_seed(42)
        for Nbatch, batch in enumerate(dataloader):
            X = batch[0].to(device,non_blocking=True)
            spatial_loss,temp_loss,result_dict_temp = model.evaluate_record(X)
            if Nbatch==0:
                result_dict = result_dict_temp
            else:
                for key in result_dict.keys():
                    if isinstance(result_dict[key],torch.Tensor):
                        result_dict[key] = torch.cat((result_dict[key],result_dict_temp[key]),dim=0)
        # _,_,result_dict = model.evaluate_record2(X.reshape(1,-1,X.shape[-1]))
        # result_dict = record(model,X)

        for key in result_dict.keys():
            if isinstance(result_dict[key],torch.Tensor):
                result_dict[key] = result_dict[key].cpu().detach().numpy()
        if (cfg.dataset.name != 'CalMS21') & (cfg.dataset.name != 'Bowen'):
            if (cfg.dataset.name != 'Lorenz'):
                result_dict['As'] = data_dict['As']
                result_dict['bs'] = data_dict['bs']
            result_dict['states_x_test'] = data_dict['states_x_test']
            result_dict['states_x_train'] = data_dict['states_x']
            if (cfg.dataset.name == 'SLDS'):
                result_dict['states_z_test'] = data_dict['states_z_test']
                result_dict['states_z_train'] = data_dict['states_z']
        for key in result_dict.keys():
            if isinstance(result_dict[key],dict):
                result_dict[key] = [result_dict[key][key2] for key2 in result_dict[key].keys()]
        ##### Save Results #####
        ioh5.save(cfg.paths.log_dir/'temp_results_{}_T{:04d}.h5'.format(epoch,seq_len),result_dict)
    else: 
        result_dict = ioh5.load(cfg.paths.log_dir /'temp_results_{}_T{:04d}.h5'.format(epoch,seq_len))
        for key in result_dict.keys():
            if isinstance(result_dict[key],dict):
                result_dict[key] = [result_dict[key][key2] for key2 in result_dict[key].keys()]
    return result_dict

def random_rotation(n, theta=None,neg=False):
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * np.random.rand()

    if n == 1:
        return np.random.rand() * np.eye(1)

    if neg:
        rot = np.array([[np.cos(theta),  np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)]])
    else:
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    out = np.eye(n)
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
    return q.dot(out).dot(q.T)


def s2b(value):
    """ Parse strings to read argparse flag entries in as bool.
    
    Parameters:
    value (str): input value
    
    Returns:
    bool
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used > tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmin(memory_available)


########## Checks if path exists, if not then creates directory ##########
def check_path(basepath, path):
    ''' Created by Elliott Abe 
        Parameters:
            basepath: string or Path object for basepath
            path: string for directory or path to check if directory exists, creates if does not exist.

        Returns:
            final_path: returns datatyep Path of combined path of basepath and path
    '''
    from pathlib import Path
    if (type(basepath)==str):
        if path in basepath:
            return basepath
        elif not os.path.exists(os.path.join(basepath, path)):
            os.makedirs(os.path.join(basepath, path))
            print('Added Directory:'+ os.path.join(basepath, path))
            return Path(os.path.join(basepath, path))
        else:
            return Path(os.path.join(basepath, path))
    else:
        if path in basepath.as_posix():
            return basepath
        elif not (basepath / path).exists():
            (basepath / path).mkdir(exist_ok=True,parents=True)
            print('Added Directory:'+ (basepath / path).as_posix())
            return (basepath / path)
        else:
            return (basepath / path)



def add_colorbar(mappable,linewidth=2,location='right',**kwargs):
    ''' modified from https://supy.readthedocs.io/en/2021.3.30/_modules/supy/util/_plot.html'''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size="5%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax, drawedges=False,**kwargs)
    cbar.outline.set_linewidth(linewidth)
    plt.sca(last_axes)
    return cbar

def map_discrete_cbar(cmap,N):
    cmap = plt.get_cmap(cmap,N+1)
    bounds = np.arange(-.5,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)



def nan_helper(y):
    """ modified from: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def interp_nans(y):
    nans, x = nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y

def nanxcorr(x, y, maxlag=25):
    """ Cross correlation ignoring NaNs.
    Parameters:
    x (np.array): array of values
    y (np.array): array of values to shift, must be same length as x
    maxlag (int): number of lags to shift y prior to testing correlation (default 25)
    
    Returns:
    cc_out (np.array): cross correlation
    lags (range): lag vector
    """
    lags = range(-maxlag, maxlag)
    cc = []
    for i in range(0, len(lags)):
        # shift data
        yshift = np.roll(y, lags[i])
        # get index where values are usable in both x and yshift
        use = ~pd.isnull(x + yshift)
        # some restructuring
        x_arr = np.asarray(x, dtype=object)
        yshift_arr = np.asarray(yshift, dtype=object)
        x_use = x_arr[use]
        yshift_use = yshift_arr[use]
        # normalize
        x_use = (x_use - np.mean(x_use)) / (np.std(x_use) * len(x_use))
        yshift_use = (yshift_use - np.mean(yshift_use)) / np.std(yshift_use)
        # get correlation
        cc.append(np.correlate(x_use, yshift_use))
    cc_out = np.hstack(np.stack(cc))

    return cc_out, lags


def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()
    
def h5load(filename):
    store = pd.HDFStore(filename)
    data = store['mydata']
    metadata = store.get_storer('mydata').attrs.metadata
    return data, metadata


def recursively_load_dict_ssm(obj,parms):
    """
    Recursively laods ssm parameters for a given ssm object
    """
    
    iterator = obj.__dict__.items()
    ans = {}
    for key, item in iterator:
        if isinstance(item,(tuple)):
            obj.__dict__[key] = tuple(parms[key])
        elif isinstance(item, np.int64):
            obj.__dict__[key] = int(parms[key])
        elif isinstance(item, np.bool_):
            obj.__dict__[key] = bool(parms[key])
        elif isinstance(item, (np.ndarray, np.float64, int, float, str, bytes)):
            obj.__dict__[key] = parms[key]
        elif (isinstance(item, (object))) and ~isinstance(item,bool):
            recursively_load_dict_ssm(item,parms[key])
        else:
            raise ValueError('Cannot load %s type'%type(item))

def autocorr(x, norm=True):
    result = np.correlate(x, x, mode='full')
    if norm:         
        return result[int(result.size/2):]/np.max(result[int(result.size/2):])
    else:
        return result[int(result.size/2):]
    

def recursively_load_dict_ssm(obj,parms):
    """
    Recursively laods ssm parameters for a given ssm object
    """
    
    iterator = obj.__dict__.items()
    ans = {}
    for key, item in iterator:
        if isinstance(item,(tuple)):
            obj.__dict__[key] = tuple(parms[key])
        elif isinstance(item, np.int64):
            obj.__dict__[key] = int(parms[key])
        elif isinstance(item, np.bool_):
            obj.__dict__[key] = bool(parms[key])
        elif isinstance(item, (np.ndarray, np.float64, int, float, str, bytes)):
            obj.__dict__[key] = parms[key]
        elif (isinstance(item, (object))) and ~isinstance(item,bool):
            recursively_load_dict_ssm(item,parms[key])
        else:
            raise ValueError('Cannot load %s type'%type(item))
    
def autocorr(x, norm=True):
    result = np.correlate(x, x, mode='full')
    if norm:         
        return result[int(result.size/2):]/np.max(result[int(result.size/2):])
    else:
        return result[int(result.size/2):]
    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


def stack_data(inputs,sequence_length,overlap=None):
    temp_data = []
    if overlap is None:
        overlap = sequence_length//2
    if len(inputs.shape)==2:
        ##### (Time,Input_Size) ##### 
        while len(inputs) > sequence_length:
                temp_data.append(inputs[:sequence_length])
                inputs = inputs[overlap:]
        return torch.stack(temp_data)
    elif len(inputs.shape)==3:
        ##### (Batch,Time,Input_Size) ##### 
        while inputs.shape[1] >= sequence_length:
                temp_data.append(inputs[:,:sequence_length])
                inputs = inputs[:,overlap:]
        return torch.concat(temp_data,axis=0)
    