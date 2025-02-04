from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

import rfmotion.utils.geometry as geometry


def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def lengths_to_query_mask(lengths, lengths_z, device, max_len: int = None):
    lengths = torch.tensor(lengths,device=device)
    lengths_z = torch.tensor(lengths_z,device=device)

    index = []
    for i in range(len(lengths)):
        segments = torch.linspace(0, lengths[i] - 1, steps=lengths_z[i] + 1, device=device).long()
        centers = (segments[:-1] + segments[1:]) // 2
        index.append(centers)

    if max_len is not None and max_len > max(lengths):
        mask = torch.zeros((len(lengths), max_len), dtype=torch.bool, device=device)
    else:
        mask = torch.zeros((len(lengths), max(lengths)), dtype=torch.bool, device=device)
    for i in range(len(lengths)):
        mask[i, index[i]] = True
    
    return index, mask


def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def remove_padding(tensors, lengths):
    return [
        tensor[:tensor_length]
        for tensor, tensor_length in zip(tensors, lengths)
    ]


def nfeats_of(rottype):
    if rottype in ["rotvec", "axisangle"]:
        return 3
    elif rottype in ["rotquat", "quaternion"]:
        return 4
    elif rottype in ["rot6d", "6drot", "rotation6d"]:
        return 6
    elif rottype in ["rotmat"]:
        return 9
    else:
        return TypeError("This rotation type doesn't have features.")


def axis_angle_to(newtype, rotations):
    if newtype in ["matrix"]:
        rotations = geometry.axis_angle_to_matrix(rotations)
        return rotations
    elif newtype in ["rotmat"]:
        rotations = geometry.axis_angle_to_matrix(rotations)
        rotations = matrix_to("rotmat", rotations)
        return rotations
    elif newtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = geometry.axis_angle_to_matrix(rotations)
        rotations = matrix_to("rot6d", rotations)
        return rotations
    elif newtype in ["rotquat", "quaternion"]:
        rotations = geometry.axis_angle_to_quaternion(rotations)
        return rotations
    elif newtype in ["rotvec", "axisangle"]:
        return rotations
    else:
        raise NotImplementedError


def matrix_to(newtype, rotations):
    if newtype in ["matrix"]:
        return rotations
    if newtype in ["rotmat"]:
        rotations = rotations.reshape((*rotations.shape[:-2], 9))
        return rotations
    elif newtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = geometry.matrix_to_rotation_6d(rotations)
        return rotations
    elif newtype in ["rotquat", "quaternion"]:
        rotations = geometry.matrix_to_quaternion(rotations)
        return rotations
    elif newtype in ["rotvec", "axisangle"]:
        rotations = geometry.matrix_to_axis_angle(rotations)
        return rotations
    else:
        raise NotImplementedError


def to_matrix(oldtype, rotations):
    if oldtype in ["matrix"]:
        return rotations
    if oldtype in ["rotmat"]:
        rotations = rotations.reshape((*rotations.shape[:-2], 3, 3))
        return rotations
    elif oldtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = geometry.rotation_6d_to_matrix(rotations)
        return rotations
    elif oldtype in ["rotquat", "quaternion"]:
        rotations = geometry.quaternion_to_matrix(rotations)
        return rotations
    elif oldtype in ["rotvec", "axisangle"]:
        rotations = geometry.axis_angle_to_matrix(rotations)
        return rotations
    else:
        raise NotImplementedError


# TODO: use a real subsampler..
def subsample(num_frames, last_framerate, new_framerate):
    step = int(last_framerate / new_framerate)
    assert step >= 1
    frames = np.arange(0, num_frames, step)
    return frames


# TODO: use a real upsampler..
def upsample(motion, last_framerate, new_framerate):
    step = int(new_framerate / last_framerate)
    assert step >= 1

    # Alpha blending => interpolation
    alpha = np.linspace(0, 1, step + 1)
    last = np.einsum("l,...->l...", 1 - alpha, motion[:-1])
    new = np.einsum("l,...->l...", alpha, motion[1:])

    chuncks = (last + new)[:-1]
    output = np.concatenate(chuncks.swapaxes(1, 0))
    # Don't forget the last one
    output = np.concatenate((output, motion[[-1]]))
    return output


if __name__ == "__main__":
    motion = np.arange(105)
    submotion = motion[subsample(len(motion), 100.0, 12.5)]
    newmotion = upsample(submotion, 12.5, 100)

    print(newmotion)
