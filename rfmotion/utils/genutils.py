from copy import copy
import numpy as np
import torch
from torch import Tensor
import logging
import os
import random
from einops import rearrange
from pathlib import Path

def extract_data_path(full_path, directory_name="data"):
    """
    Slices the given path up to and including the specified directory.

    Args:
    full_path (str): The full path as a string.
    directory_name (str): The directory to slice up to (included in the result).

    Returns:
    str: The sliced path as a string up to and including the specified directory.
    """
    path = Path(full_path)
    subpath = Path()
    for part in path.parts:
        subpath /= part
        if part == directory_name:
            break

    return str(subpath)


def freeze(model) -> None:
    r"""
    Freeze all params for inference.
    """
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

# A logger for this file
log = logging.getLogger(__name__)
def to_tensor(array):
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array)

def DotDict(in_dict):
    if isinstance(in_dict, dotdict):
        return in_dict 
    out_dict = copy(in_dict)
    for k,v in out_dict.items():
        if isinstance(v,dict):
            out_dict[k] = DotDict(v)
    return dotdict(out_dict)


def dict_to_device(tensor_dict, device):
    return {k: v.to(device) for k, v in tensor_dict.items()}

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def cast_dict_to_tensors(d, device="cpu"):
    if isinstance(d, dict):
        return {k: cast_dict_to_tensors(v, device) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        return torch.from_numpy(d).float().to(device)
    elif isinstance(d, torch.Tensor):
        return d.to(device)
    else:
        return d