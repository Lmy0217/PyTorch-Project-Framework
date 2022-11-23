from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os
import configs


__all__ = ['Summary']


class Summary(object):

    def __init__(self, path, dataset=None):
        self.summary_path = os.path.join(path, configs.env.paths.tensorboard_folder)
        self.dataset = dataset
        self.summary = self._setsummary()
        self.epoch_info = None

    def _setsummary(self):
        global _utils_summary
        if '_utils_summary' in globals() and _utils_summary is not None:
            _utils_summary.close()
        _utils_summary = SummaryWriter(self.summary_path) if configs.env.log.summary else None
        return _utils_summary

    def writer(self):
        if not configs.env.log.summary:
            import warnings
            warnings.warn('Summary is set to off.')
        return self.summary

    def norm(self, data, ms_type):
        if self.dataset is not None and hasattr(self.dataset, 'cfg') and self.dataset.cfg.norm is not None and ms_type:
            data = self.dataset.norm(data, ms_type)
        return data

    def renorm(self, data, ms_type):
        if self.dataset is not None and hasattr(self.dataset, 'cfg') and self.dataset.cfg.norm is not None and ms_type:
            data = self.dataset.renorm(data, ms_type)
        return data

    def update_epochinfo(self, epoch_info):
        self.epoch_info = epoch_info

    def add_first_sample(self, fn_str, *args, **kwargs):
        assert self.epoch_info
        if self.epoch_info['batch_idx'] == 0:
            fn = getattr(self, fn_str, None)
            if fn is not None:
                fn(*args, global_step=self.epoch_info['epoch'], **kwargs)

    def add_scalar(self, *args, **kwargs):
        if self.summary is not None:
            self.summary.add_scalar(*args, **kwargs)

    def add_scalars(self, *args, **kwargs):
        if self.summary is not None:
            self.summary.add_scalars(*args, **kwargs)

    def add_graph(self, *args, **kwargs):
        if self.summary is not None:
            self.summary.add_graph(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.summary is not None:
            self.summary.add_image(*args, **kwargs)

    def add_images(self, *args, **kwargs):
        if self.summary is not None:
            self.summary.add_images(*args, **kwargs)

    def add_vector(self, main_tag, vector, global_step=None, walltime=None):
        if self.summary is not None:
            scalars = dict()
            for id in range(len(vector)):
                scalars[str(id)] = vector[id]
            self.summary.add_scalars(main_tag, scalars, global_step, walltime)

    def add_gray2jet(self, tag, gray_tensor, ms_type='', trans=False, global_step=None, walltime=None):
        if self.summary is not None:
            if self.dataset is not None:
                gray_tensor = self.renorm(gray_tensor, ms_type)
                # cfg = getattr(self.dataset.cfg, ms_type, None)
                # value_min, value_max = getattr(cfg, 'lower', None), getattr(cfg, 'upper', None)
                # value_range = (value_min, value_max) if value_min is not None and value_max is not None else None
            if trans:
                gray_tensor = gray_tensor.permute([0, 1, 3, 2])
            gray_tensor = make_grid(gray_tensor.reshape((gray_tensor.size(1), 1, gray_tensor.size(2),
                                                         gray_tensor.size(3))), normalize=True, range=None)
            self.summary.add_image(tag, gray_tensor, global_step, walltime)
