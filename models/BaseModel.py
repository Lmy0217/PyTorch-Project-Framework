import abc
import os

import torch
import torch.nn as nn
from torch import distributed
from torch.utils.data import DataLoader

import configs
import models
import utils

__all__ = ['BaseModel']


class _MainHook(object):

    def process_pre_hook(self):
        pass

    def process_hook(self):
        pass

    def process_msg_hook(self, msg: dict):
        msg.update(dict(while_flag=False))

    def process_test_msg_hook(self, msg: dict):
        msg.update(dict(test_flag=False))


class _ProcessHook(object, metaclass=abc.ABCMeta):

    def train_pre_hook(self, epoch_info: dict, sample_dict: dict):
        return sample_dict

    @abc.abstractmethod
    def train(self, epoch_info: dict, sample_dict: dict):
        raise NotImplementedError

    def train_hook(self, epoch_info: dict, return_dict: dict):
        return return_dict

    def train_process(self, epoch_info: dict, sample_dict: dict):
        return self.train_hook(epoch_info, self.train(epoch_info, self.train_pre_hook(epoch_info, sample_dict)))

    def train_loader_hook(self, train_loader: DataLoader):
        return train_loader

    def train_epoch_pre_hook(self, epoch_info: dict, train_loader: DataLoader):
        pass

    def train_epoch_hook(self, epoch_info: dict, train_loader: DataLoader):
        pass

    def train_return_hook(self, epoch_info: dict, return_all: dict):
        return return_all

    def test_pre_hook(self, epoch_info: dict, sample_dict: dict):
        return sample_dict

    @abc.abstractmethod
    def test(self, epoch_info: dict, sample_dict: dict):
        raise NotImplementedError

    def test_hook(self, epoch_info: dict, return_dict: dict):
        return return_dict

    def test_process(self, epoch_info: dict, sample_dict: dict):
        return self.test_hook(epoch_info, self.test(epoch_info, self.test_pre_hook(epoch_info, sample_dict)))

    def test_loader_hook(self, test_loader: DataLoader):
        return test_loader

    def test_epoch_pre_hook(self, epoch_info: dict, test_loader: DataLoader):
        pass

    def test_epoch_hook(self, epoch_info: dict, test_loader: DataLoader):
        pass

    def test_return_hook(self, epoch_info: dict, return_all: dict):
        return return_all


class BaseModel(_ProcessHook, _MainHook, metaclass=abc.ABCMeta):

    logger: utils.Logger
    summary: utils.Summary
    main_msg: dict

    def __init__(self, cfg, data_cfg, run, **kwargs):
        self.name = os.path.splitext(os.path.split(cfg._path)[1])[0]
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.run = run
        self.path = utils.path.get_path(cfg, data_cfg, run)
        self.device = self.run.device

        self._save_list = []
        self.msg = {}

        for k, v in kwargs.items():
            setattr(self, k, v)

        if hasattr(self, 'summary') and hasattr(self.summary, 'dataset'):
            self.dataset = self.summary.dataset

    @staticmethod
    def check_cfg(data_cfg, cfg):
        return True

    def apply(self, fn):
        for name, value in self.__dict__.items():
            if isinstance(value, nn.Module):
                self.__dict__[name].apply(fn)

    def modules(self):
        m = {}
        for name, value in list(vars(self).items()):
            if isinstance(value, nn.Module):
                m[name] = value
        return m

    def train_return_hook(self, epoch_info: dict, return_all: dict):
        _count = torch.tensor(return_all.pop('_count'), dtype=torch.float32, device=self.device)
        _count_sum = torch.sum(_count)
        for key, value in return_all.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32, device=self.device)
            elif value.device != self.device:
                value = value.to(self.device)
            return_all[key] = _count @ value / _count_sum
        return return_all

    def summary_models(self, shapes):
        # TODO only one graph
        if hasattr(self, 'summary'):
            for name, value in self.__dict__.items():
                if isinstance(value, nn.Module):
                    self.summary.add_graph(value, torch.randn((1, *shapes[name]), device=self.device))

    def load(self, start_epoch=None, path=None):
        assert start_epoch is None or (isinstance(start_epoch, int) and start_epoch >= 0)
        path = path or self.path
        if start_epoch is None:
            main_msg = ('_' + str(self.main_msg['while_idx'])) if self.main_msg['while_idx'] > 1 else ''
            check_path = os.path.join(path, self.name + main_msg + configs.env.paths.check_file)
            if os.path.exists(check_path):
                check_data = torch.load(check_path)
                start_epoch = check_data['epoch']
                self.msg = check_data['msg']
                self.main_msg = check_data['main_msg']
            else:
                start_epoch = 0
        if start_epoch > 0:
            msg = ('_' + '-'.join(self.msg.values())) if self.msg else ''
            for name, value in self.__dict__.items():
                if isinstance(value, (nn.Module, torch.optim.Optimizer)) or name in self._save_list:
                    load_path = os.path.join(path, self.name + '_' + name + '_' + str(start_epoch) + msg + '.pth')
                    if not os.path.exists(load_path) and isinstance(value, torch.optim.Optimizer):
                        self.logger.info(f"IGNORE! Optimizer weight `{load_path}` not found!")
                        continue
                    map_location = {'cuda:%d' % 0: 'cuda:%d' % self.run.local_rank} if self.run.distributed else self.device
                    load_value = torch.load(load_path, map_location=map_location)
                    if isinstance(value, (nn.Module, torch.optim.Optimizer)):
                        self.__dict__[name].load_state_dict(load_value)
                    else:
                        self.__dict__[name] = load_value
        return start_epoch

    def save(self, epoch, path=None):
        if not self.run.distributed or (self.run.distributed and self.run.local_rank == 0):
            path = path or self.path
            if not os.path.exists(path):
                os.makedirs(path)
            msg = ('_' + '-'.join(self.msg.values())) if self.msg else ''
            for name, value in self.__dict__.items():
                # TODO remove criterion, change criterion super object to `torch.nn.modules.loss._Loss`?
                if isinstance(value, (nn.Module, torch.optim.Optimizer)) or name in self._save_list:
                    save_value = value.state_dict() if isinstance(value, (nn.Module, torch.optim.Optimizer)) else value
                    torch.save(save_value, os.path.join(path, self.name + '_' + name + '_' + str(epoch) + msg + '.pth'))
            main_msg = ('_' + str(self.main_msg['while_idx'])) if self.main_msg['while_idx'] > 1 else ''
            torch.save(dict(epoch=epoch, msg=self.msg, main_msg=self.main_msg),
                       os.path.join(path, self.name + main_msg + configs.env.paths.check_file))
        if self.run.distributed:
            torch.distributed.barrier()


if __name__ == "__main__":
    print(models.functional.common.all())
    print(models.functional.common.allcfgs())
