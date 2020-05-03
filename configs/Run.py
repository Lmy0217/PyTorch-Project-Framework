import configs
import torch


__all__ = ['Run']


class Run(configs.BaseConfig):

    def __init__(self, cfg):
        super(Run, self).__init__(cfg)
        self._more()

    def _more(self):
        self.cuda = torch.cuda.is_available() and getattr(self, 'cuda', True)
        torch.backends.cudnn.benchmark = torch.backends.cudnn.is_available()
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def all():
        return configs.all(Run, configs.env.getdir(configs.env.paths.run_cfgs_folder))


if __name__ == "__main__":
    print(Run.all())
