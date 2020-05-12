from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import abc
import os
import configs
import models
import utils


class _MainHook(object):

    def process_pre_hook(self):
        pass

    def process_hook(self):
        pass

    def process_msg_hook(self, msg: dict):
        msg.update(dict(while_flag=False))

    def process_test_msg_hook(self, msg: dict):
        msg.update((dict(test_flag=False)))


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


class BaseModel(_ProcessHook, _MainHook):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        self.name = os.path.splitext(os.path.split(cfg._path)[1])[0]
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.run = run
        self.path = utils.path.get_path(cfg, data_cfg, run)
        self.device = torch.device("cuda" if self.run.cuda else "cpu")

        self._save_list = list()
        self.msg = dict()

        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def check_cfg(data_cfg, cfg):
        return True

    def apply(self, fn):
        for name, value in self.__dict__.items():
            if value.__class__.__base__ == nn.Module:
                self.__dict__[name].apply(fn)

    def modules(self):
        m = dict()
        for name, value in list(vars(self).items()):
            if value.__class__.__base__ == nn.Module:
                m[name] = value
        return m

    def train_return_hook(self, epoch_info: dict, return_all: dict):
        _count = torch.tensor(return_all.pop('_count'), dtype=torch.float32, device=self.device)
        for key, value in return_all.items():
            return_all[key] = _count @ torch.tensor(value, dtype=torch.float32, device=self.device) \
                              / epoch_info['count_data']
        return return_all

    def summary_models(self, shapes):
        # TODO only one graph
        if hasattr(self, 'summary'):
            for name, value in self.__dict__.items():
                if value.__class__.__base__ == nn.Module:
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
                if value.__class__.__base__ == nn.Module or name in self._save_list:
                    load_value = torch.load(
                        os.path.join(path, self.name + '_' + name + '_' + str(start_epoch) + msg + '.pth'))
                    if value.__class__.__base__ == nn.Module:
                        self.__dict__[name].load_state_dict(load_value)
                    else:
                        self.__dict__[name] = load_value
        return start_epoch

    def save(self, epoch, path=None):
        path = path or self.path
        if not os.path.exists(path):
            os.makedirs(path)
        msg = ('_' + '-'.join(self.msg.values())) if self.msg else ''
        for name, value in self.__dict__.items():
            # TODO remove criterion, change criterion super object to `torch.nn.modules.loss._Loss`?
            if value.__class__.__base__ == nn.Module or name in self._save_list:
                save_value = value.state_dict() if value.__class__.__base__ == nn.Module else value
                torch.save(save_value, os.path.join(path, self.name + '_' + name + '_' + str(epoch) + msg + '.pth'))
        main_msg = ('_' + str(self.main_msg['while_idx'])) if self.main_msg['while_idx'] > 1 else ''
        torch.save(dict(epoch=epoch, msg=self.msg, main_msg=self.main_msg),
                   os.path.join(path, self.name + main_msg + configs.env.paths.check_file))


if __name__ == "__main__":
    print(models.all())
    print(models.allcfgs())
