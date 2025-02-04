import os
from omegaconf import DictConfig
import logging
import hydra

import os
import json
from omegaconf import DictConfig, OmegaConf


def save_config(cfg: DictConfig) -> str:
    path = os.path.join(cfg.run_dir, "config.json")
    config = OmegaConf.to_container(cfg, resolve=True)
    with open(path, "w") as f:
        string = json.dumps(config, indent=4)
        f.write(string)
    return path


def read_config(run_dir: str, return_json=False) -> DictConfig:
    path = os.path.join(run_dir, "config.json")
    with open(path, "r") as f:
        config = json.load(f)
    if return_json:
        return config
    cfg = OmegaConf.create(config)
    cfg.run_dir = run_dir
    return cfg

logger = logging.getLogger(__name__)


# split the lightning checkpoint into
# seperate state_dict modules for faster loading
def extract_ckpt(run_dir, ckpt_name="last"):
    import torch

    ckpt_path = os.path.join(run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")

    extracted_path = os.path.join(run_dir, f"{ckpt_name}_weights")
    os.makedirs(extracted_path, exist_ok=True)

    new_path_template = os.path.join(extracted_path, "{}.pt")
    ckpt_dict = torch.load(ckpt_path)
    state_dict = ckpt_dict["state_dict"]
    module_names = list(set([x.split(".")[0] for x in state_dict.keys()]))

    # should be ['motion_encoder', 'text_encoder', 'motion_decoder'] for example
    for module_name in module_names:
        path = new_path_template.format(module_name)
        sub_state_dict = {
            ".".join(x.split(".")[1:]): y.cpu()
            for x, y in state_dict.items()
            if x.split(".")[0] == module_name
        }
        torch.save(sub_state_dict, path)


def load_model(run_dir, **params):
    # Load last config
    cfg = read_config(run_dir)
    cfg.run_dir = run_dir
    return load_model_from_cfg(cfg, **params)


def load_model_from_cfg(cfg, ckpt_name="last", device="cpu", eval_mode=True):
    import torch

    run_dir = cfg.run_dir

    from omegaconf import DictConfig, OmegaConf

    def replace_model_with_tmr(d):
        if isinstance(d, DictConfig):
            new_dict = {}
            for k, v in d.items():
                new_key = k.replace('.model.', '.tmr.')
                new_value = replace_model_with_tmr(v)
                new_dict[new_key] = new_value
            return OmegaConf.create(new_dict)
        elif isinstance(d, dict):
            new_dict = {}
            for k, v in d.items():
                new_key = k.replace('.model.', '.tmr.')
                new_value = replace_model_with_tmr(v)
                new_dict[new_key] = new_value
            return new_dict
        elif isinstance(d, list):
            return [replace_model_with_tmr(item) for item in d]
        elif isinstance(d, str):
            return d.replace('.model.', '.tmr.')
        else:
            return d
    model_conf = replace_model_with_tmr(cfg.model)
    # import ipdb;ipdb.set_trace()

    # model_conf = OmegaConf.to_yaml(model_conf_d)
    model = hydra.utils.instantiate(model_conf)

    # Loading modules one by one
    # motion_encoder / text_encoder / text_decoder
    pt_path = os.path.join(run_dir, f"{ckpt_name}_weights")

    if not os.path.exists(pt_path):
        print("The extracted model is not found. Split into submodules..")
        extract_ckpt(run_dir, ckpt_name)

    for fname in os.listdir(pt_path):
        module_name, ext = os.path.splitext(fname)
        if ext != ".pt":
            continue

        module = getattr(model, module_name, None)
        if module is None:
            continue

        module_path = os.path.join(pt_path, fname)
        state_dict = torch.load(module_path)
        module.load_state_dict(state_dict)

    print("Loading previous checkpoint done")
    model = model.to(device)
    if eval_mode:
        model = model.eval()
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="load_model")
def hydra_load_model(cfg: DictConfig) -> None:
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt
    device = cfg.device
    eval_mode = cfg.eval_mode
    return load_model(run_dir, ckpt_name, device, eval_mode)


if __name__ == "__main__":
    hydra_load_model()
