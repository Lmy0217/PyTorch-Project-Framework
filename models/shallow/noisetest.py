import torch
import configs
import models


class NoiseTest(models.BaseModel):
    """TODO test"""

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)
        assert hasattr(cfg, 'input_list')
        assert hasattr(self.cfg, 'noise_times')
        self.noise_test = configs.BaseConfig(dict(
            name=self.__class__.__name__,
            times=self.cfg.noise_times,
            flag=False
        ))

    def _noise(self, data):
        return data + torch.randn_like(data) * 0.1 / 3

    def test_pre_hook(self, epoch_info: dict, sample_dict: dict):
        if epoch_info['epoch'] == (self.run.epochs - self.run.epochs % self.run.save_step) \
                or self.main_msg['only_test'] or self.main_msg['ci'] is not None:
            self.noise_test.flag = True
        if self.main_msg['test_idx'] > 1:
            for input_name in self.cfg.input_list:
                sample_dict[input_name] = self._noise(sample_dict[input_name])
            self.msg['NoiseTest'] = 'Noise' + str(self.main_msg['test_idx'])
        return sample_dict

    def process_test_msg_hook(self, msg: dict):
        if self.noise_test.flag:
            if msg['test_idx'] == self.noise_test.times:
                msg['test_flag'] = False
            else:
                msg['test_idx'] += 1
        else:
            msg['test_flag'] = False
