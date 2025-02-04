import importlib


def get_model(cfg, datamodule):
    modeltype = cfg.model.model_type
    model_module = importlib.import_module(f".modeltype.{modeltype}", package="rfmotion.models")
    Model = model_module.__getattribute__(f"{modeltype.upper()}")
    return Model(cfg=cfg, datamodule=datamodule)
