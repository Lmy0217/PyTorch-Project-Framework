from .BaseModel import *
from .BaseTest import *
from .shallow import *
from .layers import *
from .optimizers import *

from .wgan_gp import WGAN_GP


def all():
    return models.BaseModel.__subclasses__()


def allcfgs():
    return configs.all(configs.BaseConfig, configs.env.getdir(configs.env.paths.model_cfgs_folder))


def find(name):
    model = getattr(models, name, None)
    return model if model is not None and model.__base__ == BaseModel else None
