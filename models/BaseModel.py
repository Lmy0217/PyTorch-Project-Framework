import torch.nn as nn
import torch
import os
import configs
import models


class BaseModel(object):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.run = run
        self.device = torch.device("cuda" if self.run.cuda else "cpu")

    @staticmethod
    def check_cfg(data_cfg, cfg):
        return True

    def apply(self, fn):
        v = self.__dict__.items()
        for name, value in v:
            if value.__class__.__base__ == nn.Module:
                self.__dict__[name].apply(fn)

    def modules(self):
        m = dict()
        v = vars(self)
        for name, value in list(v.items()):
            if value.__class__.__base__ == nn.Module:
                m[name] = value
        return m

    def train(self, batch_idx, sample_dict):
        return NotImplementedError

    def test(self, batch_idx, sample_dict):
        return NotImplementedError

    def getpath(self):
        dirname = os.path.splitext(os.path.split(self.cfg._path)[1])[0] + '-' + \
                  os.path.splitext(os.path.split(self.run._path)[1])[0] + '-' + \
                  os.path.splitext(os.path.split(self.data_cfg._path)[1])[0] + '-' + str(self.data_cfg.index_cross)
        return os.path.join(configs.env.getdir(configs.env.paths.save_folder), dirname)

    def load(self, start_epoch=None, path=None, msg=None):
        assert start_epoch is None or (isinstance(start_epoch, int) and start_epoch >= 0)
        path = path or self.getpath()
        msg = ('_' + str(msg)) if msg is not None else ''
        if start_epoch is None:
            if os.path.exists(os.path.join(path, self.cfg.name + msg + configs.env.paths.check_file)):
                start_epoch = torch.load(os.path.join(path, self.cfg.name + msg + configs.env.paths.check_file))
            else:
                start_epoch = 0
        if start_epoch > 0:
            v = self.__dict__.items()
            for name, value in v:
                if value.__class__.__base__ == nn.Module:
                    self.__dict__[name].load_state_dict(torch.load(os.path.join(path, self.cfg.name + '_' + name + '_' + str(start_epoch) + msg + '.pth')))
        return start_epoch

    def save(self, epoch, path=None, msg=None):
        path = path or self.getpath()
        msg = ('_' + str(msg)) if msg is not None else ''
        if not os.path.exists(path):
            os.makedirs(path)
        v = self.__dict__.items()
        for name, value in v:
            # TODO remove criterion, change criterion super object to `torch.nn.modules.loss._Loss`?
            if value.__class__.__base__ == nn.Module:
                torch.save(value.state_dict(), os.path.join(path, self.cfg.name + '_' + name + '_' + str(epoch) + msg + '.pth'))
        torch.save(epoch, os.path.join(path, self.cfg.name + msg + configs.env.paths.check_file))


if __name__ == "__main__":
    print(models.all())
    print(models.allcfgs())
