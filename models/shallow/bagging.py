from torch.utils.data import DataLoader, Subset
import copy
import numpy as np
import os
import scipy.io
import configs
import models


class Bagging(models.BaseModel):
    """TODO test"""

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)
        assert hasattr(self.cfg, 'bag_times')
        self._save_list.append('bag')
        self.bag = configs.BaseConfig(dict(
            name=self.__class__.__name__,
            times=self.cfg.bag_times,
            size=getattr(self.cfg, 'bag_size', None),
            index=self.main_msg['while_idx']
        ))

    def _index(self, data_count):
        if self.bag.size is None:
            self.bag.size = data_count
        return np.random.randint(0, data_count, self.bag.size)

    def train_loader_hook(self, train_loader: DataLoader):
        self.msg['Bagging'] = 'bag' + str(self.main_msg['while_idx'])
        bag_dataset = Subset(train_loader.dataset, self._index(len(train_loader.dataset)))
        bag_loader = DataLoader(dataset=bag_dataset, batch_size=train_loader.batch_size, shuffle=True,
                                num_workers=train_loader.num_workers, pin_memory=train_loader.pin_memory)
        return bag_loader

    def test_return_hook(self, epoch_info: dict, return_all: dict):
        if self.main_msg['while_idx'] > 1:
            pre_msg = copy.deepcopy(self.msg)
            pre_msg['Bagging'] = 'bag' + str(self.main_msg['while_idx'] - 1)
            pre_return_all_path = os.path.join(self.path, self.name + '_' + str(epoch_info['epoch'])
                                               + '_' + '-'.join(pre_msg.values()) + configs.env.paths.predict_file)
            if self.main_msg['ci'] is None:
                pre_return_all = scipy.io.loadmat(pre_return_all_path)
                for return_name in return_all.keys():
                    return_all[return_name] = (pre_return_all[return_name] * (self.main_msg['while_idx'] - 1)
                                               + return_all[return_name]) / self.main_msg['while_idx']
        return return_all

    def process_msg_hook(self, msg: dict):
        if msg['while_idx'] == self.bag.times:
            msg['while_flag'] = False
        else:
            msg['while_idx'] += 1
