import configs
import models
import utils


__all__ = ['find', 'all', 'allcfgs']


def all():
    return utils.common.all_subclasses_not_abstract(models.BaseModel)


def allcfgs():
    return configs.all(configs.BaseConfig, configs.env.getdir(configs.env.paths.model_cfgs_folder))


def find(name):
    model = getattr(models, name, None)
    return model if model is not None and issubclass(model, models.BaseModel) else None
