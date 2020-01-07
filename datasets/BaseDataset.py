from torch.utils.data import Dataset
import torch
import numpy as np
import math
import os
import configs
import datasets


class BaseDataset(Dataset):

    def __init__(self, cfg, **kwargs):
        self.name = os.path.splitext(os.path.split(cfg._path)[1])[0]
        self.cfg = self.more(self._more(cfg))
        self.data, self.cfg.data_count = self._load()

        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def _more(cfg):
        for name, value in configs.env.dataset.dict().items():
            setattr(cfg, name, getattr(cfg, name, value))
        cfg.out = configs.BaseConfig({'elements': 1})
        return cfg

    @staticmethod
    def more(cfg):
        return cfg

    def _load(self):
        return NotImplementedError

    def _path(self, index):
        paths = self.cfg.paths
        file_count = paths.file_count if isinstance(paths.file_count, list) else [paths.file_count]
        index = index if isinstance(index, list) else [index]
        path_dict = dict()
        for name, value in vars(paths).items():
            if isinstance(value, str):
                path_dict[name] = list(value)
                if len(path_dict[name]) < 1:
                    raise ValueError
        for name, l in path_dict.items():
            l_index, l_index_start, l_index_count, l_flag = -1, -1, 0, False
            for i, d in enumerate(l):
                if d == '?':
                    if not l_flag:
                        l_index, l_index_start, l_flag = l_index + 1, i, True
                    l_index_count += 1
                else:
                    if l_flag:
                        if index[l_index] < 1 or index[l_index] > file_count[l_index] or \
                                len(str(index[l_index])) > l_index_count:
                            raise ValueError
                        for ii, ss in enumerate(list(str(index[l_index]).zfill(l_index_count))):
                            l[l_index_start + ii] = ss
                        l_index_count, l_flag = 0, False
            path_dict[name] = ''.join(l)
        return path_dict

    # TODO remove to init
    def set_logger(self, logger):
        self.logger = logger
        if hasattr(self, 'super_dataset') and isinstance(self.super_dataset, BaseDataset):
            self.super_dataset.set_logger(logger)

    @staticmethod
    def need_norm(data_shape):
        eles = 1
        for n in data_shape[1:]:
            eles *= n
        return eles > 1

    # TODO need more test
    def _ms(self, norm_set=None, split_items=None):
        assert norm_set or split_items or isinstance(split_items, dict)
        ms_dict = dict()
        ms_items = split_items[0] if split_items is not None else self.data.keys()
        for idx, name in enumerate(ms_items):
            data = self.data[name]
            if self.need_norm(data.shape):
                if norm_set is not None:
                    data = data[norm_set]
                ms_dict[name] = [np.mean(data, (0, *tuple(range(2, data.ndim)))),
                                 np.std(data, (0, *tuple(range(2, data.ndim))))]
                if split_items is not None:
                    ms_dict[split_items[1][idx]] = ms_dict[name]
        self.logger.info(ms_dict)
        return ms_dict

    def _norm(self):
        norm_func = getattr(self, '_norm_' + self.cfg.norm, None)
        if norm_func is not None:
            norm_func()
        else:
            raise NotImplementedError

    def _renorm(self):
        for name, value in self.data.items():
            if self.need_norm(value.shape):
                self.data[name] = self.renorm(value, name)

    def renorm(self, data, name, **kwargs):
        renorm_func = getattr(self, '_renorm_' + self.cfg.norm, None)
        if renorm_func is not None:
            data = renorm_func(data, name, **kwargs)
        else:
            raise NotImplementedError
        return data

    def _norm_1(self):
        for name, value in self.data.items():
            if self.need_norm(value.shape):
                ms = self.ms[-1].get(name)
                self.data[name] = \
                    (self.data[name] - np.resize(ms[0], (1, *ms[0].shape, *tuple([1] * (value.ndim - 2))))) / \
                    np.resize(ms[1], (1, *ms[1].shape, *tuple([1] * (value.ndim - 2))))

    def _renorm_1(self, data, name, ms_slice=None):
        if self.need_norm(data.shape):
            ms = self.ms[-1].get(name)
            if ms_slice is not None:
                ms = (ms[0][ms_slice], ms[1][ms_slice])
            if ms is not None:
                mean = np.resize(ms[0], (1, *ms[0].shape, *tuple([1] * (data.ndim - 2))))
                std = np.resize(ms[1], (1, *ms[1].shape, *tuple([1] * (data.ndim - 2))))
                if isinstance(data, torch.Tensor):
                    mean, std = torch.from_numpy(mean).to(data.device), torch.from_numpy(std).to(data.device)
                data = data * std + mean
        return data

    def _norm_3(self):
        for name, value in self.data.items():
            if self.need_norm(value.shape):
                ms = self.ms[-1].get(name)
                self.data[name] = \
                    (self.data[name] - np.resize(ms[0], (1, *ms[0].shape, *tuple([1] * (value.ndim - 2))))) / \
                    np.resize(ms[1], (1, *ms[1].shape, *tuple([1] * (value.ndim - 2)))) / 3

    def _renorm_3(self, data, name, ms_slice=None):
        if self.need_norm(data.shape):
            ms = self.ms[-1].get(name)
            if ms_slice is not None:
                ms = (ms[0][ms_slice], ms[1][ms_slice])
            if ms is not None:
                mean = np.resize(ms[0], (1, *ms[0].shape, *tuple([1] * (data.ndim - 2))))
                std = 3 * np.resize(ms[1], (1, *ms[1].shape, *tuple([1] * (data.ndim - 2))))
                if isinstance(data, torch.Tensor):
                    mean, std = torch.from_numpy(mean).to(data.device), torch.from_numpy(std).to(data.device)
                data = data * std + mean
        return data

    def _norm_threshold(self):
        for name, data in self.data.items():
            if self.need_norm(data.shape):
                cfg = getattr(self.cfg, name, None)
                lower, upper = getattr(cfg, 'lower', None), getattr(cfg, 'upper', None)
                if lower is not None and upper is not None:
                    self.data[name] = np.clip(self.data[name], a_min=lower, a_max=upper)
                    self.data[name] = (self.data[name] - lower) / (upper - lower)
                    self.data[name] = (self.data[name] - 0.5) / 0.5

    def _renorm_threshold(self, data, name, ms_slice=None):
        if self.need_norm(data.shape):
            cfg = getattr(self.cfg, name, None)
            lower, upper = getattr(cfg, 'lower', None), getattr(cfg, 'upper', None)
            if lower is not None and upper is not None:
                data = data * 0.5 + 0.5
                data = data * (upper - lower) + lower
        return data

    def _2dto3d(self, sample):
        assert sample.ndim == 4
        return sample[:, np.newaxis, :, :, :]

    def _recover(self, index):
        index_dict = dict()
        if hasattr(self.cfg, 'source'):
            index_dict['source'] = [0, self.cfg.source.time, 0, self.cfg.source.width, 0, self.cfg.source.height] \
                if self.cfg.source.elements > 1 else [0, self.cfg.source.elements]
        if hasattr(self.cfg, 'target'):
            index_dict['target'] = [0, self.cfg.target.time, 0, self.cfg.target.width, 0, self.cfg.target.height] \
                if self.cfg.target.elements > 1 else [0, self.cfg.target.elements]
        return [index], index_dict

    def recover(self, index):
        recovery, index_dict = self._recover(index)
        for name, value in index_dict.items():
            index_dict[name] = [slice(value[i], value[i + 1]) for i in range(0, len(value), 2)]
            index_dict[name].insert(0, recovery[0])
            index_dict[name] = tuple(index_dict[name])
        return index_dict

    def __getitem__(self, index):
        index_dict = self.recover(index)
        sample_dict = dict()
        for name, i in index_dict.items():
            sample_dict[name] = self.data[name][i]
            if getattr(self.cfg, '2dto3d', False):
                sample_dict[name] = self._2dto3d(sample_dict[name])
        return sample_dict, index

    def __len__(self):
        return self.cfg.data_count * self.cfg.out.elements

    def _cross(self, index_cross, data_count=None, elements_per_data=None):
        assert 0 <= index_cross <= self.cfg.cross_folder
        if index_cross == 0:
            return 0, (data_count or self.cfg.data_count) * (elements_per_data or self.cfg.out.elements)
        fold_length = math.floor((data_count or self.cfg.data_count) / self.cfg.cross_folder)
        fold_residual = (data_count or self.cfg.data_count) - self.cfg.cross_folder * fold_length
        adding_step = fold_residual / self.cfg.cross_folder
        adding_count = math.floor((index_cross - 1) * adding_step)
        fold_start = (index_cross - 1) * fold_length + adding_count
        fold_length += math.floor(index_cross * adding_step) - adding_count

        index_start = fold_start * (elements_per_data or self.cfg.out.elements)
        index_length = fold_length * (elements_per_data or self.cfg.out.elements)
        index_range_trainset = [[0, index_start], [index_start + index_length, len(self)]]
        index_range_testset = [[index_start, index_start + index_length]]

        norm_start = fold_start * (elements_per_data or 1)
        norm_length = fold_length * (elements_per_data or 1)
        norm_range = [[0, norm_start], [norm_start + norm_length, self.cfg.data_count]]

        return index_range_trainset, index_range_testset, norm_range, \
               [index_start, index_length, norm_start, norm_length]

    # TODO need more test
    def _reset_norm(self, norm_range=None, split_items=None):
        if self.cfg.norm:
            norm_set = None
            if norm_range is not None:
                norm_set = list()
                for nr in norm_range:
                    norm_set.extend(range(nr[0], nr[1]))
            if not hasattr(self, 'ms'):
                self.ms = list()
            else:
                self._renorm()
            self.ms.append(self._ms(norm_set, split_items))
            self._norm()

    def split(self, index_cross):
        self.cfg.index_cross = index_cross
        index_range_trainset, index_range_testset, norm_range, _ = self._cross(index_cross)
        self._reset_norm(norm_range=norm_range)
        return BaseSplit(self, index_range_trainset), BaseSplit(self, index_range_testset)


class BaseSplit(Dataset):

    def __init__(self, dataset, index_range_set):
        self.dataset = dataset
        self.indexset, self.lengthset, self.offset = self._index(index_range_set)
        self.count = len(self.indexset)
        self.raw_count = self.count // self.dataset.cfg.out.elements
        self.set_logger(self.dataset.logger)

    def _index(self, index_range_set):
        indexset, lengthset, offset, off = list(), list(), list(), 0
        for index_range in index_range_set:
            indexset.extend(range(index_range[0], index_range[1]))
            lengthset.append(index_range[1])
            off_0 = self.dataset._recover(index_range[0])[0][0]
            off_1 = self.dataset._recover(index_range[1])[0][0]
            offset.append(off - off_0)
            off += off_1 - off_0
        return indexset, lengthset, offset

    def recover(self, index):
        index_dict = self.dataset.recover(self.indexset[index])
        for name, value in index_dict.items():
            index_dict[name] = list(value)
            index_dict[name][0] += self.offset[[length > index for length in self.lengthset].index(True)]
            index_dict[name] = tuple(index_dict[name])
        return index_dict

    def __getitem__(self, index):
        return self.dataset[self.indexset[index]][0], index

    def __len__(self):
        return len(self.indexset)

    def set_logger(self, logger):
        self.logger = logger

if __name__ == "__main__":
    print(datasets.all())
    print(datasets.allcfgs())
