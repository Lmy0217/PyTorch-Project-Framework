import torch.nn as nn
import torch
import models


__all__ = ['WGAN_GP']


class Generator(nn.Module):

    def __init__(self, cfg, data_cfg, **kwargs):
        super(Generator, self).__init__()
        self.cfg = cfg
        self.data_cfg = data_cfg

        self.block1 = self._block(self.data_cfg.kernel.elements, 128, normalize=False)
        self.block2 = self._block(128, 256)
        self.block3 = self._block(256, 512)
        self.block4 = self._block(512, 1024)
        self.fc = nn.Linear(1024, self.data_cfg.out_kernel.elements)
        self.tanh = nn.Tanh()

    def _block(self, in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.fc(x)
        x = self.tanh(x)
        x = x.view(x.shape[0], self.data_cfg.out_kernel.kT, self.data_cfg.out_kernel.kW, self.data_cfg.out_kernel.kH)
        return x


class Discriminator(nn.Module):

    def __init__(self, cfg, data_cfg, **kwargs):
        super(Discriminator, self).__init__()
        self.cfg = cfg
        self.data_cfg = data_cfg

        self.fc1 = nn.Linear(self.data_cfg.out_kernel.elements, 512)
        self.leakyrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(512, 256)
        self.leakyrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.leakyrelu1(x)
        x = self.fc2(x)
        x = self.leakyrelu2(x)
        x = self.fc3(x)
        return x


class WGAN_GP(models.BaseModel):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super(WGAN_GP, self).__init__(cfg, data_cfg, run, **kwargs)
        self.G = Generator(self.cfg, self.data_cfg, **kwargs).to(self.device)
        self.D = Discriminator(self.cfg, self.data_cfg, **kwargs).to(self.device)
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.run.lr, betas=(self.run.b1, self.run.b2))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.run.lr, betas=(self.run.b1, self.run.b2))

    @staticmethod
    def check_cfg(data_cfg, cfg):
        return hasattr(data_cfg, 'source') and hasattr(data_cfg, 'target') and hasattr(data_cfg, 'kernel') \
               and data_cfg.kernel.kT == 21 and data_cfg.kernel.kW == 64 and data_cfg.kernel.kH == 64

    def _compute_gradient_penalty(self, real_target, fake_target):
        real_target, fake_target = real_target.to(self.device), fake_target.to(self.device)
        alpha = torch.rand((real_target.size(0), 1, 1, 1)).to(self.device)
        interpolates = (alpha * real_target + ((1 - alpha) * fake_target)).requires_grad_()
        d_interpolates = self.D(interpolates)
        fake_label = torch.ones(real_target.shape[0], 1).to(self.device)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake_label,
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _train_G(self, real_source, real_target):
        fake_target = self.G(real_source)
        fake_validity = self.D(fake_target)
        loss_g = -torch.mean(fake_validity)
        return loss_g

    def _train_D(self, real_source, real_target):
        fake_target = self.G(real_source)
        real_validity = self.D(real_target)
        fake_validity = self.D(fake_target)
        gradient_penalty = self._compute_gradient_penalty(real_target.detach(), fake_target.detach())
        loss_d = -torch.mean(real_validity) + torch.mean(fake_validity) + self.cfg.lambda_gp * gradient_penalty
        return loss_d

    def train(self, epoch_info, sample_dict):
        real_source, real_target = sample_dict['source'].to(self.device), sample_dict['target'].to(self.device)

        self.G.train()
        self.D.train()

        self.optimizer_D.zero_grad()
        self.loss_d = self._train_D(real_source, real_target)
        self.loss_d.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        if epoch_info['batch_idx'] % self.cfg.n_critic == 0:
            self.loss_g = self._train_G(real_source, real_target)
            self.loss_g.backward()
            self.optimizer_G.step()

        return {'loss_d': self.loss_d, 'loss_g': self.loss_g}

    def test(self, epoch_info, sample_dict):
        real_source, real_target = sample_dict['source'].to(self.device), sample_dict['target'].to(self.device)

        self.G.eval()
        self.D.eval()

        fake_target = self.G(real_source)

        return {'real_source': real_source, 'real_target': real_target,
                'fake_target': fake_target}


if __name__ == "__main__":
    models.BaseTest(WGAN_GP).run()
