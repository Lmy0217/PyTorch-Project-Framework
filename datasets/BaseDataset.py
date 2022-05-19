import abc
import math
import os
import time
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

import configs
import datasets
import utils

__all__ = ['BaseDataset', 'BaseSplit', 'SampleDataset', 'MulDataset']


class BaseDataset(Dataset, metaclass=abc.ABCMeta):

    logger: utils.Logger
    summary: utils.Summary

    def __init__(self, cfg, **kwargs):
        self.name = os.path.splitext(os.path.split(cfg._path)[1])[0]
        self.cfg = self.more(self._more(cfg))
        self.data, self.cfg.data_count = self.load()

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

    @abc.abstractmethod
    def load(self):
        raise NotImplementedError

    def _path(self, index):
        paths = self.cfg.paths
        file_count = paths.file_count if isinstance(paths.file_count, list) else [paths.file_count]
        index = index if isinstance(index, list) else [index]
        return utils.path.comp_path(vars(paths), file_count, index)

    # TODO remove to init
    def set_logger(self, logger):
        self.logger = logger
        if hasattr(self, 'super_dataset') and isinstance(self.super_dataset, BaseDataset):
            self.super_dataset.set_logger(logger)

    def set_summary(self, summary):
        self.summary = summary

    @staticmethod
    def need_norm(data_shape):
        for n in data_shape[1:]:
            if n > 1:
                return True
        return False

    @staticmethod
    def meanstd(data):
        return data.mean((0, *tuple(range(2, data.ndim)))), data.std((0, *tuple(range(2, data.ndim))))

    def _ms(self, norm_set: Union[None, list] = None, split_items: Union[None, list] = None):
        assert norm_set or (split_items and len(split_items) == 2 and isinstance(split_items[0], list)
                            and isinstance(split_items[1], list) and len(split_items[0]) == len(split_items[1]))
        ms_dict = dict()
        ms_items = split_items[0] if split_items is not None else self.data.keys()
        for idx, name in enumerate(ms_items):
            data = self.data[name]
            if self.need_norm(data.shape):
                if norm_set is not None:
                    data = data[norm_set]
                ms_dict[name] = self.meanstd(data)
                if split_items is not None:
                    ms_dict[split_items[1][idx]] = ms_dict[name]
        if hasattr(self, 'logger'):
            self.logger.info(ms_dict)
            self.logger.save_mat(
                self.name + configs.env.paths.ms_file,
                {
                    key: (value[0].cpu().numpy(), value[1].cpu().numpy())
                    if isinstance(value[0], torch.Tensor) else value
                    for key, value in ms_dict.items()
                }
            )
        return ms_dict

    def _norm(self):
        for name, value in self.data.items():
            self.data[name] = self.norm(value, name)

    def _renorm(self):
        for name, value in self.data.items():
            self.data[name] = self.renorm(value, name)

    def norm(self, data, name, **kwargs):
        assert self.cfg.norm is not None
        norm_func = getattr(datasets.functional.norm, self.cfg.norm + '_norm', None)
        if norm_func is not None:
            if self.need_norm(data.shape):
                data = norm_func(**self._get_args(norm_func, data, name, **kwargs))
        else:
            raise NotImplementedError('method {}_norm is not implemented!'.format(self.cfg.norm))
        return data

    def renorm(self, data, name, **kwargs):
        assert self.cfg.norm is not None
        renorm_func = getattr(datasets.functional.norm, self.cfg.norm + '_renorm', None)
        if renorm_func is not None:
            if self.need_norm(data.shape):
                data = renorm_func(**self._get_args(renorm_func, data, name, **kwargs))
        else:
            raise NotImplementedError('method {}_renorm is not implemented!'.format(self.cfg.norm))
        return data

    def _get_args(self, norm_func, data, name: str, **kwargs):
        ms = self.ms[-1].get(name)
        if ms is not None and 'ms_slice' in kwargs.keys():
            ms_slice = kwargs['ms_slice']
            ms = (ms[0][ms_slice], ms[1][ms_slice])
        cfg = getattr(self.cfg, name, getattr(self.cfg, name.split('_')[0], None))
        args_dict = dict()
        for key in norm_func.__code__.co_varnames:
            if key == 'data':
                args_dict['data'] = data
            elif key == 'mean':
                if ms is not None:
                    mean = ms[0].reshape((1, *ms[0].shape, *tuple([1] * (data.ndim - 2))))
                    if isinstance(data, torch.Tensor) and isinstance(mean, np.ndarray):
                        mean = torch.from_numpy(mean).to(data.device)
                    args_dict['mean'] = mean
            elif key == 'std':
                if ms is not None:
                    std = ms[1].reshape((1, *ms[1].shape, *tuple([1] * (data.ndim - 2))))
                    if isinstance(data, torch.Tensor) and isinstance(std, np.ndarray):
                        std = torch.from_numpy(std).to(data.device)
                    args_dict['std'] = std
            elif hasattr(cfg, key):
                args_dict[key] = getattr(cfg, key)
            else:
                raise NameError('parameter \'{}\' is not defined'.format(key))
        return args_dict

    def _recover(self, index):
        index_dict = dict()
        for key, value in self.cfg.dict().items():
            if key not in ['out', 'kernel'] and isinstance(value, configs.BaseConfig):
                if hasattr(value, 'width') and hasattr(value, 'height'):
                    if hasattr(value, 'time'):
                        if hasattr(value, 'patch'):
                            index_dict[key] = [0, value.patch, 0, value.time, 0, value.width, 0, value.height]
                        else:
                            index_dict[key] = [0, value.time, 0, value.width, 0, value.height]
                    else:
                        index_dict[key] = [0, value.width, 0, value.height]
                elif hasattr(value, 'elements'):
                    index_dict[key] = [0, value.elements]
        return [index], index_dict

    def recover(self, index):
        recovery, index_dict = self._recover(index)
        for name, value in index_dict.items():
            index_dict[name] = [slice(value[i], value[i + 1]) for i in range(0, len(value), 2)]
            index_dict[name].insert(0, recovery[0])
            index_dict[name] = tuple(index_dict[name])
        return index_dict

    def __getitem__(self, index):
        with torch.no_grad():
            index_dict = self.recover(index)
            sample_dict = dict()
            for name, i in index_dict.items():
                sample_dict[name] = self.data[name][i]
        return sample_dict, index

    def __len__(self):
        return self.cfg.data_count * self.cfg.out.elements

    def _cross(self, index_cross: int, data_count=None, elements_per_data=None):
        # Using: data_count as the number of classes and elements_per_data as the number of images in each class
        assert 0 < index_cross <= self.cfg.cross_folder or (index_cross == 0 and self.cfg.cross_folder == 0)
        if self.cfg.cross_folder == 0:
            return [[0, len(self)]], [], [[0, self.cfg.data_count]], [0, 0, 0, 0]
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

    def _reset_norm(self, norm_range: Union[None, list] = None, split_items: Union[None, list] = None):
        norm_set = None
        if norm_range is not None:
            norm_set = list()
            for nr in norm_range:
                assert isinstance(nr, list) and len(nr) == 2
                norm_set.extend(range(nr[0], nr[1]))
        self.split_set = norm_set
        self.split_item = split_items

        if self.cfg.norm is not None:
            if norm_set or split_items:
                if not hasattr(self, 'ms'):
                    self.ms = list()
                else:
                    self._renorm()
                self.ms.append(self._ms(norm_set, split_items))
                self._norm()
            else:
                self.cfg.norm = None
                if hasattr(self, 'logger'):
                    self.logger.info('Normset is empty! Setting cfg.norm = None')

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
        if hasattr(self.dataset, 'logger'):
            self.set_logger(self.dataset.logger)
        if hasattr(self.dataset, 'summary'):
            self.set_summary(self.dataset.summary)

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
        return self.count

    def set_logger(self, logger):
        self.logger = logger

    def set_summary(self, summary):
        self.summary = summary


class SampleDataset(Dataset):

    def __init__(self, data: dict):
        self.data = data

    def __getitem__(self, index):
        item = dict()
        for key, value in self.data.items():
            item[key] = value[index]
        return item, index

    def __len__(self):
        return len(list(self.data.values())[0])


class MulDataset(BaseDataset):

    @staticmethod
    def _more(cfg):
        cfg = BaseDataset._more(cfg)
        cfg.num_workers = np.inf
        cfg.pin_memory = True
        cfg.cross_folder = 0
        for idx in range(len(cfg.cfgs)):
            if not isinstance(cfg.cfgs[idx], configs.BaseConfig):
                cfg.cfgs[idx] = configs.BaseConfig(utils.path.real_config_path(cfg.cfgs[idx], configs.env.paths.dataset_cfgs_folder))
            cfg.cfgs[idx] = datasets.functional.common.more(cfg.cfgs[idx])
            if cfg.cfgs[idx].num_workers < cfg.num_workers:
                cfg.num_workers = cfg.cfgs[idx].num_workers
            cfg.pin_memory &= cfg.cfgs[idx].pin_memory
        cfg.count_cfgs = len(cfg.cfgs)
        return cfg

    def load(self):
        ds = []
        data_count = 0
        for idx in range(self.cfg.count_cfgs):
            d = datasets.functional.common.find(self.cfg.cfgs[idx].name)(self.cfg.cfgs[idx])
            data_count += len(d)
            ds.append(d)
        return {'datasets': ds}, data_count

    def split(self, index_cross=None):
        self.trainsets, self.testsets = [], []
        self.trainset_length, self.testset_length = [], []
        for idx in range(self.cfg.count_cfgs):
            self.data['datasets'][idx].set_logger(self.logger)
            self.data['datasets'][idx].set_summary(self.summary)
            trainset, testset = self.data['datasets'][idx].split(index_cross)
            self.trainsets.append(trainset)
            self.testsets.append(testset)
            self.trainset_length.append(len(trainset))
            self.testset_length.append(len(testset))
        self.trainset_cumsum, self.testset_cumsum = np.cumsum(self.trainset_length), np.cumsum(self.testset_length)
        self.trainset_length, self.testset_length = np.sum(self.trainset_length), np.sum(self.testset_length)
        self.cfg.data_count = self.trainset_length + self.testset_length
        index_range_trainset = [[0, self.trainset_length]]
        index_range_testset = [[self.trainset_length, self.cfg.data_count]]
        return datasets.BaseSplit(self, index_range_trainset), datasets.BaseSplit(self, index_range_testset)

    def get_idx(self, index):
        if not hasattr(self, 'flag_seed'):
            self.flag_seed = None
        if index < self.trainset_length:
            idx_dataset = np.sum(index >= self.trainset_cumsum)
            idx = (index - self.trainset_cumsum[idx_dataset - 1]) if idx_dataset > 0 else index
            if self.flag_seed is None or self.flag_seed == 'test':
                utils.common.set_seed(int(time.time() + index))
                self.flag_seed = 'train'
                for i in range(self.cfg.count_cfgs):
                    self.data['datasets'][i].flag_seed = self.flag_seed
        else:
            index = index - self.trainset_length
            idx_dataset = np.sum(index >= self.testset_cumsum)
            idx = (index - self.testset_cumsum[idx_dataset - 1]) if idx_dataset > 0 else index
            utils.common.set_seed(index * 3)
            self.flag_seed = 'test'
            for i in range(self.cfg.count_cfgs):
                self.data['datasets'][i].flag_seed = self.flag_seed
        return idx_dataset, idx

    def __getitem__(self, index):
        idx_dataset, idx = self.get_idx(index)
        list_dataset = self.trainsets if self.flag_seed == 'train' else self.testsets
        # print(index, self.flag_seed, idx_dataset, idx)
        item = list_dataset[idx_dataset][idx]
        # TODO: nested overlays
        item[0]['idx_dataset'] = torch.tensor([idx_dataset])
        item[0]['idx_sample'] = torch.tensor([idx])
        return item


if __name__ == "__main__":
    print(datasets.functional.common.all())
    print(datasets.functional.common.allcfgs())
