from typing import Any, Dict, List
import argparse
import os
import copy
import torch
from clip import clip
import pickle
from functools import partial
import os.path as osp


class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

def print_color(text, color):
    print(color + text + Colors.RESET)
    
    
def load_clip_to_cpu():
    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def clip_clipping(x):
    #! -inf ~ inf -> CLIP's input RGB range
    if len(x.shape) == 3:
        out = torch.cat([torch.clip(x[0,:,:], min=-1.79226253, max=1.93033625).unsqueeze(0),
                     torch.clip(x[1,:,:], min=-1.75209713, max=2.07488384).unsqueeze(0),
                     torch.clip(x[2,:,:], min=-1.48021977, max=2.14589699).unsqueeze(0)], dim=0)
    else:
        out = torch.cat([torch.clip(x[:,0,:,:], min=-1.79226253, max=1.93033625).unsqueeze(1),
                        torch.clip(x[:,1,:,:], min=-1.75209713, max=2.07488384).unsqueeze(1),
                        torch.clip(x[:,2,:,:], min=-1.48021977, max=2.14589699).unsqueeze(1)], dim=1)
    return out

def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint


def save_checkpoint(save_file_path, epoch, model):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
        
    save_states = {
        'epoch': epoch,
        'state_dict': model_state_dict
    }
    torch.save(save_states, save_file_path)


def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../datasets/")

    parser.add_argument("--non_iid", type=int, default=1)  # 0: IID, 1: Non-IID
    parser.add_argument("--n_clients", type=int, default=100)
    parser.add_argument("--n_shards", type=int, default=200)
    parser.add_argument("--frac", type=float, default=0.1)

    parser.add_argument("--n_commmunication_rounds", type=int, default=1000)
    parser.add_argument("--n_client_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="/home/mikael/Code/black-box-fed-avg/log")
    parser.add_argument("--save_dir", type=str, default="/home/mikael/Code/black-box-fed-avg/result")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="exp")

    return parser.parse_args()
