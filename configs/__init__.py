import os

from .BaseConfig import *
from .BaseTest import *

from .Env import *
from .Run import *


def all(config, cfg_dir):
    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)
    cfg_list = list()
    for file in sorted(os.listdir(cfg_dir)):
        cfg_list.append(config(os.path.join(cfg_dir, file)))
    return cfg_list
