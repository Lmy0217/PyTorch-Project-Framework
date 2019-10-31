from .BaseDataset import *
from .BaseTest import *


def more(cfg):
    dataset = getattr(datasets, cfg.name, None)
    return dataset.more(dataset._more(cfg)) if dataset else cfg


def allcfgs():
    return [more(cfg) for cfg
            in configs.all(configs.BaseConfig, configs.env.getdir(configs.env.paths.dataset_cfgs_folder))]


def all():
    return datasets.BaseDataset.__subclasses__()


def find(name):
    dataset = getattr(datasets, name, None)
    return dataset if dataset is not None and dataset.__base__ == BaseDataset else None
