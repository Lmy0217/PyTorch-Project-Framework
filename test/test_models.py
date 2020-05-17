import unittest
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import BaseConfig, Run, env
from models import BaseModel
from models.functional.algorithm import correct


__all__ = ['TestBaseModel', 'TestFunctional_algorithm']


class TestBaseModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(env.getdir(env.paths.test_folder),
                                os.path.splitext(os.path.basename(__file__))[0], cls.__name__)
        if not os.path.exists(cls.path):
            os.makedirs(cls.path)

        cls.old_save_folder = env.paths.save_folder
        env.paths.save_folder = os.path.join(env.paths.test_folder, cls.path)

        class SimpleModel(BaseModel):

            def __init__(self, cfg, data_cfg, run, **kwargs):
                super().__init__(cfg, data_cfg, run, **kwargs)
                self.fc = nn.Linear(6, 3).to(self.device)

            def train(self, epoch_info, sample_dict):
                input = sample_dict['input'].to(self.device)
                target = sample_dict['target'].to(self.device)
                output = self.fc(input)
                loss = F.l1_loss(output, target)
                return {'loss': loss}

            def test(self, batch_idx, sample_dict):
                input = sample_dict['input'].to(self.device)
                output = self.fc(input)
                return {'output': output}

        model_path = os.path.join(cls.path, 'model_configs.json')
        with open(model_path, 'w') as f:
            f.write(json.dumps(dict(name='SimpleModel')))

        dataset_path = os.path.join(cls.path, 'dataset_configs.json')
        with open(dataset_path, 'w') as f:
            f.write(json.dumps(dict(index_cross=1)))

        run_path = os.path.join(cls.path, 'run_configs.json')
        with open(run_path, 'w') as f:
            f.write(json.dumps(dict()))

        cls.model = SimpleModel(BaseConfig(model_path), BaseConfig(dataset_path), Run(run_path))

    @classmethod
    def tearDownClass(cls):
        env.paths.save_folder = cls.old_save_folder

    def testInit(self):
        self.assertEqual(self.model.name, 'model_configs')
        self.assertEqual(self.model.device, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def test_apply(self):
        def weight_init(m):
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0)
        self.model.apply(weight_init)
        self.assertTrue((self.model.fc.weight == 1.0).all())
        self.assertTrue((self.model.fc.bias == 0).all())

    def test_modules(self):
        modules = self.model.modules()
        self.assertTrue(isinstance(modules['fc'], nn.Linear))

    def test_save_load(self):
        value_weight = torch.rand(1).item()
        value_bias = torch.rand(1).item()
        self.model.fc.weight.data.fill_(value_weight)
        self.model.fc.bias.data.fill_(value_bias)
        self.model.main_msg = dict(while_idx=1)
        self.model.save(1, path=self.model.path)
        start_epoch = self.model.load(path=self.model.path)
        self.assertEqual(start_epoch, 1)
        self.assertTrue((self.model.fc.weight == value_weight).all())
        self.assertTrue((self.model.fc.bias == value_bias).all())


class TestFunctional_algorithm(unittest.TestCase):

    def test_correct(self):
        validity = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        label = torch.tensor([3])
        fake_label, accuracy = correct(validity, label, softmax=False)
        self.assertTrue((fake_label == torch.tensor([[3]])).all())
        self.assertEqual(accuracy, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
