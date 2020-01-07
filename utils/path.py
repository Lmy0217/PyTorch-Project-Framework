import os
import configs


def get_filename(path):
    return os.path.splitext(os.path.split(path)[1])[0]


def get_path(model_cfg, dataset_cfg, run_cfg):
    dirname = get_filename(model_cfg._path) + '-' + get_filename(run_cfg._path) + '-' + \
              get_filename(dataset_cfg._path) + '-' + str(dataset_cfg.index_cross)
    return os.path.join(configs.env.getdir(configs.env.paths.save_folder), dirname)
