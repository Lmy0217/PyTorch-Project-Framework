import configs
import datasets
import utils


__all__ = ['more', 'all', 'allcfgs', 'find']


def more(cfg):
    dataset = getattr(datasets, cfg.name, None)
    return dataset.more(dataset._more(cfg)) if dataset else cfg


def allcfgs():
    return [more(cfg) for cfg
            in configs.all(configs.BaseConfig, configs.env.getdir(configs.env.paths.dataset_cfgs_folder))]


def get_cfg(cfg_name):
    return more(configs.BaseConfig(utils.path.real_config_path(cfg_name, configs.env.paths.dataset_cfgs_folder)))


def all():
    return utils.common.all_subclasses_not_abstract(datasets.BaseDataset)


def find(name):
    dataset = getattr(datasets, name, None)
    return dataset if dataset is not None and issubclass(dataset, datasets.BaseDataset) else None


def get_dataset(cfg):
    return find(cfg.name)(cfg)


def get(cfg_name):
    return get_dataset(get_cfg(cfg_name))
