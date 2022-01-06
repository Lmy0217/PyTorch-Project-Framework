import argparse
import os
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import configs
import datasets
import models
import utils


class Main(object):

    def __init__(self, args):
        self.args = args
        self.model_cfg = configs.BaseConfig(utils.path.real_config_path(
            args.model_config_path, configs.env.paths.model_cfgs_folder))
        self.run_cfg = configs.Run(utils.path.real_config_path(
            args.run_config_path, configs.env.paths.run_cfgs_folder), gpus=args.gpus)
        self.dataset_cfg = datasets.functional.common.more(configs.BaseConfig(
            utils.path.real_config_path(args.dataset_config_path, configs.env.paths.dataset_cfgs_folder)))

        if not self.run_cfg.distributed or (self.run_cfg.distributed and self.run_cfg.local_rank == 0):
            print(args)

        self._init()
        self._get_component()

    def _init(self):
        utils.common.set_seed(0)
        configs.env.ci.run = self.args.ci
        # TODO remove msg['ci'], use configs.env.ci.run
        self.msg = dict(ci='ci' if configs.env.ci.run else None)

    def _get_component(self):
        self.dataset = datasets.functional.common.find(self.dataset_cfg.name)(self.dataset_cfg)

    def show_cfgs(self):
        self.logger.info(self.model.cfg)
        self.logger.info(self.run_cfg)
        self.logger.info(self.dataset.cfg)

    def split(self, index_cross):
        self.dataset_cfg.index_cross = index_cross
        self.path = utils.path.get_path(self.model_cfg, self.dataset_cfg, self.run_cfg)

        self.logger = utils.Logger(self.path, utils.path.get_filename(self.model_cfg._path))
        self.dataset.set_logger(self.logger)
        self.summary = utils.Summary(self.path, dataset=self.dataset)
        self.dataset.set_summary(self.summary)

        self.trainset, self.testset = self.dataset.split(index_cross)

        train_sampler = torch.utils.data.distributed.DistributedSampler(self.trainset, shuffle=True) \
            if self.run_cfg.distributed else None
        test_sampler = torch.utils.data.distributed.DistributedSampler(self.testset, shuffle=False) \
            if self.run_cfg.distributed else None
        # more than one num_workers, use screen to detach
        # TODO fix num_workers must set 0 in Windows
        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.run_cfg.dist_batchsize if self.run_cfg.distributed else self.run_cfg.batch_size,
            shuffle=(train_sampler is None),
            collate_fn=getattr(self.trainset.dataset, 'collate_fn', None),
            num_workers=0 if platform.system() == 'Windows' else self.dataset.cfg.num_workers,
            pin_memory=self.dataset.cfg.pin_memory,
            sampler=train_sampler
        ) if len(self.trainset) > 0 else list()

        self.run_cfg.test_batch_size = getattr(self.run_cfg, 'test_batch_size', self.run_cfg.batch_size)
        self.test_loader = DataLoader(
            self.testset,
            batch_size=self.run_cfg.dist_batchsize if self.run_cfg.distributed else self.run_cfg.test_batch_size,
            shuffle=False,
            collate_fn=getattr(self.testset.dataset, 'collate_fn', None),
            num_workers=0 if platform.system() == 'Windows' else self.dataset.cfg.num_workers,
            pin_memory=self.dataset.cfg.pin_memory,
            sampler=test_sampler
        ) if len(self.testset) > 0 else list()

        self.model = models.functional.common.find(self.model_cfg.name)(
            self.model_cfg, self.dataset.cfg, self.run_cfg, logger=self.logger, summary=self.summary, main_msg=self.msg)
        self.start_epoch = self.model.load(self.args.test_epoch)

        if not self.run_cfg.distributed or (self.run_cfg.distributed and self.run_cfg.local_rank == 0):
            self.show_cfgs()

    @staticmethod
    def _get_type(cfg: configs.BaseConfig, data_name: str, test: bool = True):
        data_type = data_name.split('_')[-1]
        data_cfg = getattr(cfg, 'test_' + data_type, getattr(cfg, data_type, None)) \
            if test else getattr(cfg, data_type, None)
        if data_cfg is None and len(data_name.split('_')) > 1:
            data_type = data_name.split('_')[-2] + '_' + data_name.split('_')[-1]
            data_cfg = getattr(cfg, 'test_' + data_type, getattr(cfg, data_type, None)) \
                if test else getattr(cfg, data_type, None)
        return data_type, data_cfg

    def train(self, epoch):
        utils.common.set_seed(int(time.time()))
        torch.cuda.empty_cache()
        count, loss_all = 0, dict()
        self.train_loader = self.model.train_loader_hook(self.train_loader)
        batch_per_epoch, count_data = len(self.train_loader), len(self.train_loader.dataset)
        log_step = max(int(np.power(10, np.floor(np.log10(batch_per_epoch / 10)))), 1) if batch_per_epoch > 0 else 1
        epoch_info = {'epoch': epoch, 'batch_per_epoch': batch_per_epoch, 'count_data': count_data}
        self.model.train_epoch_pre_hook(epoch_info, self.train_loader)
        for batch_idx, (sample_dict, index) in enumerate(self.train_loader):
            _count = len(list(sample_dict.values())[0])
            epoch_info['batch_idx'] = batch_idx
            epoch_info['index'] = index
            epoch_info['batch_count'] = _count
            self.summary.update_epochinfo(epoch_info)
            loss_dict = self.model.train_process(epoch_info, sample_dict)
            loss_dict.update(dict(_count=_count))
            utils.common.merge_dict(loss_all, loss_dict)
            count += _count
            # TODO more scalars?
            # self.summary.add_scalars('Losses', loss_dict, (epoch - 1) * batch_per_epoch + batch_idx + 1)
            if batch_idx % log_step == 0:
                if self.run_cfg.distributed:
                    count_rank = (count - _count) * self.run_cfg.world_size + _count * (self.run_cfg.local_rank + 1)
                    with utils.ddp.sequence():
                        self.logger.info_scalars(
                            'Train Epoch: {} rank {} [{}/{} ({:.0f}%)]\t',
                            (epoch, self.run_cfg.local_rank, count_rank, count_data, 100. * count_rank / count_data),
                            loss_dict
                        )
                else:
                    # TODO error when `dataset` has `collate_fn`, item include batch and `count_data` is count of batch,
                    #      but `count` is all of items.
                    self.logger.info_scalars('Train Epoch: {} [{}/{} ({:.0f}%)]\t',
                                             (epoch, count, count_data, 100. * count / count_data), loss_dict)
        self.model.train_epoch_hook(epoch_info, self.train_loader)
        if epoch % self.run_cfg.save_step == 0:
            loss_file = os.path.join(self.path, self.model.name + '_' + str(epoch)
                                     + (('_' + '-'.join(self.model.msg.values())) if self.model.msg else '')
                                     + configs.env.paths.loss_file)
            self.logger.save_mat(
                loss_file, {k: v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for k, v in loss_all.items()})
        loss_all = self.model.train_return_hook(epoch_info, loss_all)
        if self.run_cfg.distributed:
            with utils.ddp.sequence():
                self.logger.info_scalars('Train Epoch: {} rank {}\t', (epoch, self.run_cfg.local_rank), loss_all)
        else:
            self.logger.info_scalars('Train Epoch: {}\t', (epoch,), loss_all)
        if epoch % self.run_cfg.save_step == 0:
            self.model.save(epoch)

    # TODO simplify
    def test(self, epoch):
        utils.common.set_seed(int(time.time()))
        torch.cuda.empty_cache()
        predict = dict()
        count = 0
        add_data_msgs, msgs, msgs_dict = None, None, dict()
        with torch.no_grad():
            self.test_loader = self.model.test_loader_hook(self.test_loader)
            batch_per_epoch, count_data = len(self.test_loader), len(self.test_loader.dataset)
            log_step = max(int(np.power(10, np.floor(np.log10(batch_per_epoch / 10)))), 1) if batch_per_epoch > 0 else 1
            epoch_info = {'epoch': epoch, 'batch_per_epoch': batch_per_epoch, 'count_data': count_data}
            self.model.test_epoch_pre_hook(epoch_info, self.test_loader)
            for batch_idx, (sample_dict, index) in enumerate(self.test_loader):
                _count = len(list(sample_dict.values())[0])
                epoch_info['batch_idx'] = batch_idx
                epoch_info['index'] = index
                epoch_info['batch_count'] = _count
                self.summary.update_epochinfo(epoch_info)
                output_dict = self.model.test_process(epoch_info, sample_dict)
                # TODO remove msgs (Diagnosis), change diagnosis datasets cfgs filename
                if isinstance(output_dict, tuple):
                    if len(output_dict) == 2:
                        add_data_msgs, output_dict = output_dict[1], output_dict[0]
                    elif len(output_dict) == 3:
                        msgs, add_data_msgs, output_dict = output_dict[2], output_dict[1], output_dict[0]
                count += _count
                if msgs is not None:
                    for name, value in msgs.items():
                        if name not in msgs_dict.keys():
                            msgs_dict[name] = value
                        else:
                            msgs_dict[name] += value
                if batch_idx % log_step == 0:
                    if self.run_cfg.distributed:
                        count_rank = (count - _count) * self.run_cfg.world_size + _count * (self.run_cfg.local_rank + 1)
                        with utils.ddp.sequence():
                            self.logger.info('Test Epoch: {} rank {} [{}/{} ({:.0f}%)]'.format(
                                epoch, self.run_cfg.local_rank, count_rank, count_data, 100. * count_rank / count_data))
                    else:
                        self.logger.info('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                            epoch, count, count_data, 100. * count / count_data))
                if len(predict) != len(output_dict):
                    self.predict_device = torch.device(
                        'cuda' if self.model.run.cuda and self.dataset.cfg.predict_cuda else 'cpu') \
                        if self.dataset.cfg.predict_cuda is not None else None
                    for name, value in output_dict.items():
                        data_type, data_cfg = self._get_type(self.dataset.cfg, name, test=True)
                        # TODO check data_cfg has value but not BaseConfig
                        if data_cfg is not None:
                            if add_data_msgs is None:
                                if data_cfg.elements > 1:
                                    if hasattr(data_cfg, 'patch'):
                                        predict_shape = (data_cfg.patch, data_cfg.time, data_cfg.width, data_cfg.height)
                                    elif hasattr(data_cfg, 'time'):
                                        predict_shape = (data_cfg.time, data_cfg.width, data_cfg.height)
                                    elif hasattr(data_cfg, 'width'):
                                        predict_shape = (data_cfg.width, data_cfg.height)
                                    else:
                                        predict_shape = [data_cfg.elements]
                                else:
                                    predict_shape = [1]
                            else:
                                predict_shape = value.shape[1:]
                            predict[name] = torch.zeros(
                                self.testset.raw_count, *predict_shape, dtype=torch.float32,
                                device=self.predict_device or value.device)
                        else:
                            predict[name] = torch.tensor(
                                [], dtype=torch.float32, device=self.predict_device or value.device)
                for name, value in output_dict.items():
                    if self.predict_device is not None:
                        value = value.to(self.predict_device)
                    data_type, data_cfg = self._get_type(self.dataset.cfg, name, test=True)
                    if data_cfg is not None:
                        for i in range(len(value)):
                            slice_recover = self.testset.recover(index[i])[data_type]
                            if add_data_msgs is not None:
                                slice_recover = slice_recover[0]
                            predict[name][slice_recover] = value[i]
                    else:
                        predict[name] = torch.cat((predict[name], value.float()
                        if value.shape else torch.tensor([value], dtype=torch.float32,
                                                         device=self.predict_device or value.device)))
            self.model.test_epoch_hook(epoch_info, self.test_loader)
            if msgs is not None:
                log_msg = 'Test Epoch: {}'
                accuracy = list()
                for name, value in msgs_dict.items():
                    log_msg += ' ' + name + ': {:0.2f}%'
                    msgs_dict[name] = 100. * value / count
                    accuracy.append(msgs_dict[name])
                self.logger.info(log_msg.format(epoch, *accuracy))
                self.summary.add_scalars('Accuracy', msgs_dict, epoch)

        # TODO do not support chain norm and renorm
        dataset_cfg, dataset = [self.dataset.cfg], [self.dataset]
        # while hasattr(dataset_cfg[-1], 'super'):
        #     dataset.append(dataset[-1].super_dataset)
        #     dataset_cfg.append(dataset[-1].cfg)
        for name, value in predict.items():
            predict[name] = np.array(value.cpu())
            for d_cfg, d in zip(dataset_cfg, dataset):
                if d_cfg.norm and d.need_norm(value.shape):
                    data_type, data_cfg = self._get_type(d_cfg, name, test=False)
                    if data_cfg is not None:
                        other = dict()
                        if add_data_msgs is not None:
                            key = name.split('_')[0] + '_' + name.split('_')[1]
                            if key in add_data_msgs.keys():
                                one_msg = add_data_msgs[key]
                                one_shape = getattr(one_msg, data_type)
                                ms_slice = slice(one_shape.bT, one_shape.bT + one_shape.time)
                                other['ms_slice'] = ms_slice
                        predict[name] = d.renorm(predict[name], data_type, **other)
        predict = self.model.test_return_hook(epoch_info, predict)
        predict_file = os.path.join(self.path, self.model.name + '_' + str(epoch)
                                    + (('_' + '-'.join(self.model.msg.values())) if self.model.msg else '')
                                    + configs.env.paths.predict_file)
        self.logger.save_mat(predict_file, predict)


def run():
    parser = argparse.ArgumentParser(description='Template')
    parser.add_argument('-m', '--model_config_path', type=str, required=True, metavar='/path/to/model/config.json',
                        help='Path to model config .json file')
    parser.add_argument('-r', '--run_config_path', type=str, required=True, metavar='/path/to/run/config.json',
                        help='Path to run config .json file')
    parser.add_argument('-d', '--dataset_config_path', type=str, required=True, metavar='/path/to/dataset/config.json',
                        help='Path to dataset config .json file')
    parser.add_argument('-g', '--gpus', type=str, default='0', metavar='cuda device, i.e. 0 or 0,1,2,3 or cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('-t', '--test_epoch', type=int, metavar='epoch want to test', help='epoch want to test')
    parser.add_argument('--ci', action='store_false' if configs.env.ci.run else 'store_true',
                        default=configs.env.ci.run, help='running CI')
    args = parser.parse_args()

    main = Main(args)
    for index_cross in range(min(main.dataset.cfg.cross_folder, 1), main.dataset.cfg.cross_folder + 1):
        main.msg.update(dict(index_cross=index_cross, while_idx=1, while_flag=True))
        while main.msg['while_flag']:
            main.split(index_cross)
            main.model.process_pre_hook()
            if args.test_epoch is None:
                if main.start_epoch == 0:
                    main.test(main.start_epoch)
                for epoch in range(main.start_epoch + 1, main.run_cfg.epochs + 1):
                    main.train(epoch)
                    if epoch % main.run_cfg.save_step == 0:
                        main.model.main_msg.update(dict(test_idx=1, test_flag=True, only_test=False))
                        while main.model.main_msg['test_flag']:
                            main.test(epoch)
                            main.model.process_test_msg_hook(main.model.main_msg)
            else:
                main.model.main_msg.update(dict(test_idx=1, test_flag=True, only_test=True))
                while main.model.main_msg['test_flag']:
                    main.test(main.start_epoch)
                    main.model.process_test_msg_hook(main.model.main_msg)
            main.model.process_hook()
            main.model.process_msg_hook(main.msg)


if __name__ == '__main__':
    run()
