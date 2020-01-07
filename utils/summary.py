from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os
import configs


class Summary(object):

    def __init__(self, path, dataset=None):
        self.summary_path = os.path.join(path, configs.env.paths.tensorboard_folder)
        self.dataset = dataset
        self.summary = self._setsummary()

    def _setsummary(self):
        global _utils_summary
        if '_utils_summary' in globals():
            _utils_summary.close()
        _utils_summary = SummaryWriter(self.summary_path)
        return _utils_summary

    def writer(self):
        return self.summary

    def add_scalar(self, *args, **kwargs):
        self.summary.add_scalar(*args, **kwargs)

    def add_scalars(self, *args, **kwargs):
        self.summary.add_scalars(*args, **kwargs)

    def add_graph(self, *args, **kwargs):
        self.summary.add_graph(*args, **kwargs)

    def add_gray2jet(self, tag, gray_tensor, ms_type='', trans=False, global_step=None, walltime=None):
        if self.dataset is not None:
            gray_tensor = self.dataset.renorm(gray_tensor.detach(), ms_type)
            cfg = getattr(self.dataset.cfg, ms_type, None)
            value_min, value_max = getattr(cfg, 'lower', None), getattr(cfg, 'upper', None)
            value_range = (value_min, value_max) if value_min is not None and value_max is not None else None
        if trans:
            gray_tensor = gray_tensor.permute([0, 1, 3, 2])
        gray_tensor = make_grid(gray_tensor.reshape((gray_tensor.size(1), 1, gray_tensor.size(2), gray_tensor.size(3))),
                                normalize=True, range=None)
        self.summary.add_image(tag, gray_tensor, global_step, walltime)
