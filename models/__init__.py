from .BaseModel import *
from .BaseTest import *

from . import functional
from . import shallow
from . import layers
from . import optimizers

from .wgan_gp import WGAN_GP
from .lenet import LeNet

import utils


def all():
    return utils.common.all_subclasses_not_abstract(BaseModel)


def allcfgs():
    return configs.all(configs.BaseConfig, configs.env.getdir(configs.env.paths.model_cfgs_folder))


def find(name):
    model = getattr(models, name, None)
    return model if model is not None and issubclass(model, BaseModel) else None
