import clip
import importlib

from rfmotion.models.architectures.motionclip.transformer import Encoder_TRANSFORMER, Decoder_TRANSFORMER
from rfmotion.models.architectures.motionclip.motionclip import MOTIONCLIP

JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]

LOSSES = ["rc", "rcxyz", "vel", "velxyz"]  # not used: "hp", "mmd", "vel", "velxyz"

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # assert len(unexpected_keys) == 0
    # assert all([k.startswith('clip_model.') for k in missing_keys])

def get_model(parameters, clip_model):
    encoder = Encoder_TRANSFORMER(**parameters)
    decoder = Decoder_TRANSFORMER(**parameters)
    parameters["outputxyz"] = "rcxyz" in parameters["lambdas"]
    return MOTIONCLIP(encoder, decoder, clip_model=clip_model, **parameters).to(parameters["device"])

def get_model_and_data(parameters, split="train"):

    # clip_model, preprocess = clip.load("ViT-B/32", device=device)  # Must set jit=False for training
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=parameters['device'], jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16

    for domain in parameters.get('clip_training', '').split('_'):
        clip_num_layers = parameters.get('clip_layers', 12)
        if domain == 'text':
            clip_model.initialize_parameters()
            clip_model.transformer.resblocks = clip_model.transformer.resblocks[:clip_num_layers]
        if domain == 'image':
            clip_model.initialize_parameters()
            clip_model.visual.transformer = clip_model.transformer.resblocks[:clip_num_layers]

    # NO Clip Training ,Freeze CLIP weights
    if parameters.get('clip_training', '') == '':
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

    # datasets = get_datasets(parameters, clip_preprocess, split)
    datasets = None
    # parameters["num_classes"] = len([0]) # dummy_class = [0]
    # parameters["nfeats"] = 6 #  rot6d  self.nfeats
    # parameters["njoints"] = 25 # smpl 25 joints self.njoints 
    model = get_model(parameters, clip_model)
    return model