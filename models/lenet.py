import torch.nn as nn
import torch
import models

__all__ = ['LeNet']


class Structure(nn.Module):

    def __init__(self, cfg, data_cfg, **kwargs):
        super(Structure, self).__init__()
        self.cfg = cfg
        self.data_cfg = data_cfg

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet(models.BaseModel):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super(LeNet, self).__init__(cfg, data_cfg, run, **kwargs)
        self.structure = Structure(self.cfg, self.data_cfg, **kwargs).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.structure.parameters(), lr=self.run.lr, betas=self.run.betas)

    @staticmethod
    def check_cfg(data_cfg, cfg):
        return data_cfg.name == 'MNIST'

    def train(self, epoch_info, sample_dict):
        real_source, real_target = sample_dict['source'].to(self.device), sample_dict['target'].to(self.device)

        self.structure.train()

        self.optimizer.zero_grad()
        fake_validity = self.structure(real_source)
        loss = self.criterion(fake_validity, real_target.reshape(real_target.size(0)))
        loss.backward()
        self.optimizer.step()

        accuracy = models.functional.algorithm.correct(fake_validity, real_target, return_label=False) / real_target.size(0)

        return {'loss': loss, 'accuracy': torch.tensor(accuracy)}

    def test(self, epoch_info, sample_dict):
        real_source, real_target = sample_dict['source'].to(self.device), sample_dict['target'].to(self.device)

        self.structure.eval()

        fake_validity = self.structure(real_source)
        loss = self.criterion(fake_validity, real_target.reshape(real_target.size(0)))

        fake_target, accuracy = models.functional.algorithm.correct(fake_validity, real_target)

        return {'real_source': real_source, 'real_target': real_target, 'fake_target': fake_target,
                'loss': loss, 'accuracy': torch.tensor(accuracy)}, {}, {'accuracy': accuracy}


if __name__ == "__main__":
    models.BaseTest(LeNet).run()
