import torch
from torch.utils.data import default_collate
from typing import List, Dict, Optional
from torch import Tensor
def length_to_mask(length, device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask

def collate_batch_last_padding(batch, feats):
    t2m = True
    for batch_items in batch:
        for bk in batch_items.keys():
            if 'source' in bk:
                t2m = False
    if not t2m:
        feats_src = [f'{featype}_source' for featype in feats]
        feats_tgt = [f'{featype}_target' for featype in feats]
        tot_feats = feats_src + feats_tgt
    else:
        feats_tgt = [f'{featype}_target' for featype in feats]
        tot_feats = feats_tgt
        t2m = True

    # this should be used only when we mix hml3d with other datasets
    # check if we need to duplicate the _target features as _source features
    # for hml3d in case of not t2m
    if not t2m:
        for b in batch:
            if b['dataset_name'] == 'hml3d':
                keys = list(b.keys())
                for k in keys:
                    if k.endswith('_target'):
                        if k == 'length_target':
                            b[k.replace('target', 'source')] = 0
                        else:
                            b[k.replace('target', 'source')] = torch.zeros_like(b[k])

    batch = pad_batch(batch, tot_feats, t2m=t2m)
    batch =  {k: torch.stack([b[k] for b in batch])\
              if k in tot_feats or k.endswith('_norm') else [b[k] for b in batch]
              for k in batch[0].keys()}
    
    return batch

def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def pad_batch(batch, feats, t2m):
    """
    pad feature tensors to account for different number of frames
    we do NOT zero pad to avoid wierd values in normalisation later in the model
    input:
        - batch: list of input dictionaries
        - feats: list of features to apply padding on (rest left as is)
    returns:
        - padded batch list
        - original length of sequences (could be < n_frames when subsequensing)
    """
    if t2m:
        max_frames = max(len(b['body_pose_target']) for b in batch)
        pad_length = torch.tensor([max_frames - len(b['body_pose_target'])
                                for b in batch])
        collated_batch =  [{k: _apply_on_feats(v, k, _pad_n(pad_length[i]), 
                                               feats)
                            for k, v in b.items()}
                            for i, b in enumerate(batch)]
        for x in collated_batch:
            x['task'] = 't2m'
    else:
        max_frames_src = max(len(b['body_pose_source']) for b in batch)
        max_frames_tgt = max(len(b['body_pose_target']) for b in batch)
        pad_length_src = torch.tensor([max_frames_src - len(b['body_pose_source'])
                                for b in batch])
        pad_length_tgt = torch.tensor([max_frames_tgt - len(b['body_pose_target'])
                                for b in batch])
        collated_batch =  [{k: _apply_on_feats(v, k, _pad_n(pad_length_src[i]), feats)
            if '_source' in k else _apply_on_feats(v, k, _pad_n(pad_length_tgt[i]), feats)
                for k, v in b.items()}
            for i, b in enumerate(batch)]
        for x in collated_batch:
            x['task'] = 'edit'
    return collated_batch

def _pad_n(n):
    """get padding function for padding x at the first dimension n times"""
    from torch.nn.functional import pad
    return lambda x: pad(x[None], (0, 0) * (len(x.shape) - 1) + (0, n), "replicate")[0]

def _apply_on_feats(t, name: str, f, feats):
    """apply function f only on features"""
    return f(t) if name in feats or name.endswith('_norm') else t

def collate_x_dict(lst_x_dict: List, *, device: Optional[str] = None) -> Dict:
    x = collate_tensor_with_padding([x_dict["x"] for x_dict in lst_x_dict])
    if device is not None:
        x = x.to(device)
    length = [x_dict["length"] for x_dict in lst_x_dict]
    mask = length_to_mask(length, device=x.device)
    batch = {"x": x, "length": length, "mask": mask}
    return batch


def collate_text_motion(lst_elements: List, *, device: Optional[str] = None) -> Dict:
    one_el = lst_elements[0]
    keys = one_el.keys()

    x_dict_keys = [key for key in keys if "x_dict" in key]
    other_keys = [key for key in keys if "x_dict" not in key]

    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and device is not None:
            batch[key] = val.to(device)

    for key in x_dict_keys:
        batch[key] = collate_x_dict([x[key] for x in lst_elements], device=device)
    return batch

def collate_text_and_body_parts(lst_elements: List[Dict]) -> Dict:
    text_keys = [
        key for key in lst_elements[0] if isinstance(lst_elements[0][key], str)
    ]

    batch = {key: [x[key] for x in lst_elements] for key in text_keys}

    bp_gpt = collate_tensor_with_padding(
        [torch.tensor(x["bp-gpt"]) for x in lst_elements])

    batch["bp_gpt"] = bp_gpt
    return batch

def collate_datastruct_and_text(lst_elements: List) -> Dict:
    # collate_datastruct = lst_elements[0]["datastruct"].transforms.collate
    keys_not_tensor = ['n_frames_orig', 'framerate', 'length_source',
                       'length_target', 'text', 'filename', 'split',
                       'id']
    # keys_tensor = [k for k in lst_elements[0].keys() if k not in keys_not_tensor]
                
    batch = {# Collate with padding for the datastruct
             k : pad_batch([x[k] for x in lst_elements])\
             if k not in keys_not_tensor
             else [x[k] for x in lst_elements]
             for k in lst_elements[0].keys() }
    # add keyid for example
    # otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    # for key in otherkeys:
    #     batch[key] = [x[key] for x in lst_elements]

    return batch


def collate_length_and_text(lst_elements: List) -> Dict:

    batch = {
        "length_0": [x["length_0"] for x in lst_elements],
        "length_1": [x["length_1"] for x in lst_elements],
        "length_transition": [x["length_transition"] for x in lst_elements],
        "length_1_with_transition":
        [x["length_1_with_transition"] for x in lst_elements],
        "text_0": [x["text_0"] for x in lst_elements],
        "text_1": [x["text_1"] for x in lst_elements]
    }

    return batch


def collate_pairs_and_text(lst_elements: List, ) -> Dict:
    if 'features_0' not in lst_elements[0]:  # test set
        collate_datastruct = lst_elements[0]["datastruct"].transforms.collate
        batch = {
            "datastruct":
            collate_datastruct([x["datastruct"] for x in lst_elements]),
            "length_0": [x["length_0"] for x in lst_elements],
            "length_1": [x["length_1"] for x in lst_elements],
            "length_transition":
            [x["length_transition"] for x in lst_elements],
            "length_1_with_transition":
            [x["length_1_with_transition"] for x in lst_elements],
            "text_0": [x["text_0"] for x in lst_elements],
            "text_1": [x["text_1"] for x in lst_elements]
        }

    else:
        batch = {
            "motion_feats_0":
            collate_tensor_with_padding(
                [el["features_0"] for el in lst_elements]),
            "motion_feats_1":
            collate_tensor_with_padding(
                [el["features_1"] for el in lst_elements]),
            "motion_feats_1_with_transition":
            collate_tensor_with_padding(
                [el["features_1_with_transition"] for el in lst_elements]),
            "length_0": [x["length_0"] for x in lst_elements],
            "length_1": [x["length_1"] for x in lst_elements],
            "length_transition":
            [x["length_transition"] for x in lst_elements],
            "length_1_with_transition":
            [x["length_1_with_transition"] for x in lst_elements],
            "text_0": [x["text_0"] for x in lst_elements],
            "text_1": [x["text_1"] for x in lst_elements]
        }
    return batch


def collate_text_and_length(lst_elements: Dict) -> Dict:
    batch = {
        "length": [x["length"] for x in lst_elements],
        "text": [x["text"] for x in lst_elements]
    }

    # add keyid for example
    otherkeys = [
        x for x in lst_elements[0].keys()
        if x not in batch and x != "datastruct"
    ]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]
    return batch
