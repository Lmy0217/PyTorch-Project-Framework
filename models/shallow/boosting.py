import torch
import models
import configs
import abc
import scipy.io
import numpy as np
import os

from models.shallow import NoiseTest


__all__ = ['AdaBoost_Image', 'AdaBoost_Image_NoiseTest']


class AdaBoost_Image(models.BaseModel, metaclass=abc.ABCMeta):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)
        assert hasattr(self.cfg, 'return_list')
        self._save_list.append('boost')
        self.boost = configs.BaseConfig(dict(
            name=self.__class__.__name__,
            error=list(),
            alpha=list(),
            eps=1e-20
        ))

    def train_epoch_pre_hook(self, epoch_info, train_loader):
        if epoch_info['epoch'] == 1:
            self.boost.length = epoch_info['count_data']
            self.boost.weight = torch.zeros(self.boost.length, device=self.device).fill_(1 / self.boost.length)

    def train_pre_hook(self, epoch_info, sample_dict):
        self.boost.index = epoch_info['index'].to(self.device)
        return sample_dict

    def weight_loss(self, loss):
        weight = self.boost.weight.gather(0, self.boost.index)
        return weight @ loss

    @abc.abstractmethod
    def comp_score(self, epoch_info, sample_dict):
        raise NotImplementedError

    def train_epoch_hook(self, epoch_info, train_loader):
        if epoch_info['epoch'] % self.run.save_step == 0 or self.main_msg['ci'] is not None:
            with torch.no_grad():
                self.boost.indexes = torch.tensor([], dtype=torch.long, device=self.device)
                self.boost.scores = torch.tensor([], device=self.device)
                for batch_idx, (sample_dict, index) in enumerate(train_loader):
                    self.boost.indexes = torch.cat((self.boost.indexes, index.to(self.device)))
                    self.boost.scores = torch.cat((self.boost.scores, self.comp_score(epoch_info, sample_dict)))

                error = self.boost.weight.gather(0, self.boost.indexes) @ (self.boost.scores / (self.boost.scores + 1))
                if self.main_msg['ci'] is None:
                    assert 0 < error < 0.5
                alpha = torch.log((1 - error) / error) / 2
                self.boost.error.append(error)
                self.boost.alpha.append(alpha)

                self.boost.weight *= torch.exp((self.boost.scores * 2 - 1) * alpha)  # need detach
                assert torch.sum(self.boost.weight) != 0
                self.boost.weight /= torch.sum(self.boost.weight)

    def train_return_hook(self, epoch_info, return_all):
        return_all = super().train_return_hook(epoch_info, return_all)
        if epoch_info['epoch'] % self.run.save_step == 0 or self.main_msg['ci'] is not None:
            return_all['error'] = self.boost.error[-1]
            return_all['alpha'] = self.boost.alpha[-1]
        return return_all

    def test_return_hook(self, epoch_info, return_all):
        if epoch_info['epoch'] >= 2 * self.run.save_step:
            pre_return_all_path = os.path.join(self.path, self.name + '_'
                                               + str(epoch_info['epoch'] - self.run.save_step)
                                               + (('_' + '-'.join(self.msg.values())) if self.msg else '')
                                               + configs.env.paths.predict_file)
            if self.main_msg['ci'] is None:
                pre_return_all = scipy.io.loadmat(pre_return_all_path)
                for return_name in self.cfg.return_list:
                    pre_sum_alpha = np.sum(self.boost.alpha[:-1]).item()
                    return_all[return_name] = \
                        (pre_return_all[return_name] * pre_sum_alpha + return_all[return_name]
                         * self.boost.alpha[-1].item()) / (pre_sum_alpha + self.boost.alpha[-1].item())
        return return_all


class AdaBoost_Image_NoiseTest(AdaBoost_Image, NoiseTest):

    def test_return_hook(self, epoch_info, return_all):
        if 'NoiseTest' in self.msg.keys():
            noise_msg = self.msg.pop('NoiseTest')
            return_all = super().test_return_hook(epoch_info, return_all)
            self.msg['NoiseTest'] = noise_msg
        else:
            return_all = super().test_return_hook(epoch_info, return_all)
        return return_all
