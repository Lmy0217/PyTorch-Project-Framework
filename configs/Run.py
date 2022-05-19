import os
import torch
from torch import distributed
from torch.backends import cudnn

import configs

__all__ = ['Run']


class Run(configs.BaseConfig):

    epochs: int
    save_step: int
    batch_size: int
    test_batch_size: int

    def __init__(self, cfg, gpus: str = '0', **kwargs):
        super(Run, self).__init__(cfg, gpus=gpus, **kwargs)
        self._more()

    def _more(self):
        self._set_gpus()
        if self.gpus:
            self.cuda = torch.cuda.is_available() and getattr(self, 'cuda', True)
            self.device = torch.device("cuda", 0) if self.cuda else torch.device("cpu")
        else:
            self.cuda = False
            self.device = torch.device('cpu')

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        self.distributed = self.cuda and len(self.gpus) > 1 and \
                           torch.distributed.is_available() and torch.distributed.is_nccl_available()
        if self.distributed:
            if 'LOCAL_RANK' not in os.environ:
                raise ValueError('check run `torchrun` or `python -m torch.distributed.launch --use_env`')
            self.local_rank = int(os.environ['LOCAL_RANK'])

            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)

            torch.distributed.init_process_group(backend="nccl", init_method='env://')
            self.world_size = torch.distributed.get_world_size()

            assert self.batch_size % self.world_size == 0, 'batch_size must be multiple of CUDA device count'
            self.dist_batchsize = self.batch_size // self.world_size
            assert self.test_batch_size % self.world_size == 0, 'test_batch_size must be multiple of CUDA device count'
            self.dist_test_batchsize = self.test_batch_size // self.world_size

    def _set_gpus(self):
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            self.gpus = os.environ['CUDA_VISIBLE_DEVICES']

        if self.gpus.lower() == 'cpu':
            self.gpus = []
        elif self.gpus == '':
            self.gpus = list(range(torch.cuda.device_count()))
        else:
            self.gpus = [int(g) for g in self.gpus.split(',')]

        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in self.gpus])

    @staticmethod
    def all():
        return configs.all(Run, configs.env.getdir(configs.env.paths.run_cfgs_folder))


if __name__ == "__main__":
    print(Run.all())
