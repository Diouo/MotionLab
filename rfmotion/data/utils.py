import torch


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# padding to max length in one batch
def collate_tensors(batch):
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


def all_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    notnone_batches.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = {
        "word_embs":collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "pos_ohot":collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        "text": [b[2] for b in notnone_batches],
        "text_len":collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
        "motion":collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
        "length": [b[5] for b in notnone_batches],
        "tokens": [b[6] for b in notnone_batches],

        "edit_text": [b[2] for b in notnone_batches],
        "source_motion": collate_tensors([torch.tensor(b[8]).float() for b in notnone_batches]),
        "target_motion": collate_tensors([torch.tensor(b[9]).float() for b in notnone_batches]),
        "length_source": [b[10] for b in notnone_batches],
        "length_target": [b[11] for b in notnone_batches],

        # "text2": [b[7] for b in notnone_batches],
        # "reference_motion": collate_tensors([torch.tensor(b[8]).float() for b in notnone_batches]),
        # "reference_length":collate_tensors([torch.tensor(b[9]) for b in notnone_batches]),
        # "label": [torch.tensor(int(b[10])) for b in notnone_batches],
        # "style_text": [b[11] for b in notnone_batches],

        # "edit_text": [b[12] for b in notnone_batches],
        # "source_motion": collate_tensors([torch.tensor(b[13]).float() for b in notnone_batches]),
        # "target_motion": collate_tensors([torch.tensor(b[14]).float() for b in notnone_batches]),
        # "length_source": [b[15] for b in notnone_batches],
        # "length_target": [b[16] for b in notnone_batches],
    }

    return adapted_batch

def humanml3d_100style_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    
    batch_len = len(notnone_batches[0])

    if batch_len == 3:
        is_cmld = True
        stage = "cmld"
    elif batch_len == 6:
        stage = "cmld_cycle"
    elif batch_len == 9:
        stage = "cmld_tri"
    elif batch_len == 7:
        stage = "test_walk"
    elif batch_len == 8:
        stage = "test_xia"
    elif batch_len == 12 :
        stage = "test"
    else:
        stage = "other"
    # batch.sort(key=lambda x: x[3], reverse=True)


    if stage == "other":
        notnone_batches.sort(key=lambda x: x[3], reverse=True)
        adapted_batch = {
            "motion":
            collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
            "text": [b[2] for b in notnone_batches],
            "length": [b[5] for b in notnone_batches],
            "word_embs":
            collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
            "pos_ohot":
            collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "text_len":
            collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
            "tokens": [b[6] for b in notnone_batches],
        }
    elif stage == 'test':
        notnone_batches.sort(key=lambda x: x[3], reverse=True)
        adapted_batch = {
            "word_embs":collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
            "pos_ohot":collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "text": [b[2] for b in notnone_batches],
            "text_len":collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
            "motion":collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
            "length": [b[5] for b in notnone_batches],
            "tokens": [b[6] for b in notnone_batches],

            "text2": [b[7] for b in notnone_batches],
            "reference_motion": collate_tensors([torch.tensor(b[8]).float() for b in notnone_batches]),
            "reference_length":collate_tensors([torch.tensor(b[9]) for b in notnone_batches]),
            "label": [torch.tensor(int(b[10])) for b in notnone_batches],
            "style_text": [b[11] for b in notnone_batches],
        }
    elif stage == 'test_xia':
        notnone_batches.sort(key=lambda x: x[3], reverse=True)
        adapted_batch = {
            "motion":
            collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "text": [b[0] for b in notnone_batches],
            "length": [b[2] for b in notnone_batches],

            "reference_motion": collate_tensors([torch.tensor(b[5]).float() for b in notnone_batches]),
            "text2": [b[4] for b in notnone_batches],
            "label": [torch.tensor(int(b[7])) for b in notnone_batches],
            "label_c": [torch.tensor(int(b[3])) for b in notnone_batches],
            "text_len2": [b[6] for b in notnone_batches], #collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
        }
    elif stage == 'test_walk':
        notnone_batches.sort(key=lambda x: x[3], reverse=True)

        adapted_batch = {
            "motion":
            collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "text": [b[0] for b in notnone_batches],
            "length": [b[2] for b in notnone_batches],

            "reference_motion": collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
            "text2": [b[3] for b in notnone_batches],
            "text_len2": [b[5] for b in notnone_batches], #collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
            "label": [torch.tensor(int(b[6])) for b in notnone_batches],
        }

        # adapted_batch = {
        #     "motion_1":
        #     collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        #     "text_1": [b[0] for b in notnone_batches],
        #     "length_1": [b[2] for b in notnone_batches],

        #     "motion_2": collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
        #     "text_2": [b[3] for b in notnone_batches],
        #     "length_2": [b[5] for b in notnone_batches], #collate_tensors([torch.tensor(b[5]) for b in notnone_batches]),
        #     "style_text": [b[6] for b in notnone_batches],
        # }

    elif stage == "cmld":
        notnone_batches.sort(key=lambda x: x[2], reverse=True)
        adapted_batch = {
            "motion":
            collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "text": [b[0] for b in notnone_batches],
            "length": [b[2] for b in notnone_batches],
        }
    elif stage == "cmld_tri":
        notnone_batches.sort(key=lambda x: x[2], reverse=True)
        adapted_batch = {
            "motion_1":collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "text_1": [b[0] for b in notnone_batches],
            "length_1": [b[2] for b in notnone_batches],

            "motion_1_same":collate_tensors([torch.tensor(b[7]).float() for b in notnone_batches]),
            "text_1_same": [b[6] for b in notnone_batches],
            "length_1_same": [b[8] for b in notnone_batches],
            
            "motion_2":collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
            "text_2": [b[3] for b in notnone_batches],
            "length_2": [b[5] for b in notnone_batches],
        }
    else:
        notnone_batches.sort(key=lambda x: x[2], reverse=True)
        adapted_batch = {
            "motion_1":collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
            "text_1": [b[0] for b in notnone_batches],
            "length_1": [b[2] for b in notnone_batches],
            
            "motion_2":collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
            "text_2": [b[3] for b in notnone_batches],
            "length_2": [b[5] for b in notnone_batches],
        }


    return adapted_batch

# an adapter to our collate func
def mld_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    notnone_batches.sort(key=lambda x: x[3], reverse=True)
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
        "text": [b[2] for b in notnone_batches],
        "length": [b[5] for b in notnone_batches],
        "word_embs":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "pos_ohot":
        collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        "text_len":
        collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
        "tokens": [b[6] for b in notnone_batches],
    }
    return adapted_batch

# an adapter to our collate func
def motionfix_retarget_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    notnone_batches.sort(key=lambda x: x[3], reverse=True)
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = {
        "text": [b[0] for b in notnone_batches],
        "source_motion": collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        "length_source": [b[3] for b in notnone_batches],
        "target_motion": collate_tensors([torch.tensor(b[2]).float() for b in notnone_batches]),
        "length_target": [b[4] for b in notnone_batches],
    }
    return adapted_batch


def a2m_collate(batch):

    databatch = [b[0] for b in batch]
    labelbatch = [b[1] for b in batch]
    lenbatch = [len(b[0][0][0]) for b in batch]
    labeltextbatch = [b[3] for b in batch]

    databatchTensor = collate_tensors(databatch)
    labelbatchTensor = torch.as_tensor(labelbatch).unsqueeze(1)
    lenbatchTensor = torch.as_tensor(lenbatch)

    maskbatchTensor = lengths_to_mask(lenbatchTensor)
    adapted_batch = {
        "motion": databatchTensor.permute(0, 3, 2, 1).flatten(start_dim=2),
        "action": labelbatchTensor,
        "action_text": labeltextbatch,
        "mask": maskbatchTensor,
        "length": lenbatchTensor
    }
    return adapted_batch
