import os
import configs


def get_filename(path):
    return os.path.splitext(os.path.split(path)[1])[0]


def get_path(model_cfg, dataset_cfg, run_cfg):
    dirname = get_filename(model_cfg._path) + '-' + get_filename(run_cfg._path) + '-' + \
              get_filename(dataset_cfg._path) + '-' + str(dataset_cfg.index_cross)
    return os.path.join(configs.env.getdir(configs.env.paths.save_folder), dirname)


def comp_path(paths: dict, counts: list, indexes: list):
    path_dict = dict()
    for name, value in paths.items():
        if isinstance(value, str):
            path_dict[name] = list(value)
            if len(path_dict[name]) < 1:
                raise ValueError('path is empty!')
    for name, l in path_dict.items():
        l_index, l_index_start, l_index_count, l_flag = -1, -1, 0, False
        for i, d in enumerate(l):
            if d == '?':
                if not l_flag:
                    l_index, l_index_start, l_flag = l_index + 1, i, True
                l_index_count += 1
            else:
                if l_flag:
                    if indexes[l_index] < 1 or indexes[l_index] > counts[l_index] or \
                            len(str(indexes[l_index])) > l_index_count:
                        raise ValueError('index error!')
                    for ii, ss in enumerate(list(str(indexes[l_index]).zfill(l_index_count))):
                        l[l_index_start + ii] = ss
                    l_index_count, l_flag = 0, False
        if l_flag:
            if indexes[l_index] < 1 or indexes[l_index] > counts[l_index] or \
                    len(str(indexes[l_index])) > l_index_count:
                raise ValueError('index error!')
            for ii, ss in enumerate(list(str(indexes[l_index]).zfill(l_index_count))):
                l[l_index_start + ii] = ss
        path_dict[name] = ''.join(l)
    return path_dict
