import os

from .BaseConfig import BaseConfig
from .BaseTest import BaseTest
from .Env import env
from .Run import Run

__all__ = ['BaseConfig', 'BaseTest', 'Run', 'env', 'all']


def all(config, cfg_dir):
    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)
    cfg_list = list()
    for file in sorted(os.listdir(cfg_dir)):
        cfg_list.append(config(os.path.join(cfg_dir, file)))
    return cfg_list
