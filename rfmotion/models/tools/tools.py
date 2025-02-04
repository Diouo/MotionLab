import torch
import torch.nn as nn
from rfmotion.tools.transforms3d import transform_body_pose

def remove_padding(tensors, lengths):
    return [tensor[:tensor_length] for tensor, tensor_length in zip(tensors, lengths)]

class AutoParams(nn.Module):
    def __init__(self, **kargs):
        try:
            for param in self.needed_params:
                if param in kargs:
                    setattr(self, param, kargs[param])
                else:
                    raise ValueError(f"{param} is needed.")
        except :
            pass
            
        try:
            for param, default in self.optional_params.items():
                if param in kargs and kargs[param] is not None:
                    setattr(self, param, kargs[param])
                else:
                    setattr(self, param, default)
        except :
            pass
        super().__init__()


# taken from joeynmt repo
def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def pack_to_render(rots, trans, pose_repr='6d'):
    # make axis-angle
    # global_orient = transform_body_pose(rots, f"{pose_repr}->aa")
    if pose_repr != 'aa':
        body_pose = transform_body_pose(rots, f"{pose_repr}->aa")
    else:
        body_pose = rots
    if trans is None:
        trans = torch.zeros((rots.shape[0], rots.shape[1], 3),device=rots.device)
    render_d = {'body_transl': trans,
                'body_orient': body_pose[..., :3],
                'body_pose': body_pose[..., 3:]}
    return render_d

def split_list(lst):
    length = len(lst)
    if length == 0:
        return [], None, []
    mid_index = length // 2
    front_half = lst[:mid_index]
    middle = lst[mid_index] if length % 2 != 0 else None
    back_half = lst[mid_index + 1:] if length % 2 != 0 else lst[mid_index:]
    return front_half, middle, back_half
