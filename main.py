from torch.utils.data import DataLoader
import argparse
import torch
import os
import time
import scipy.io
import numpy as np
import configs
import datasets
import models
import utils
import platform


class Main(object):

    def __init__(self, args):
        self.args = args
        self.model_cfg = configs.BaseConfig(args.model_config_path)
        self.run_cfg = configs.Run(args.run_config_path)
        self.dataset_cfg = datasets.more(configs.BaseConfig(args.dataset_config_path))

        self._init()
        self._get_component()

    def _init(self):
        torch.manual_seed(int(time.time()))
        configs.env.ci.run = self.args.ci

    def _get_component(self):
        self.dataset = datasets.find(self.dataset_cfg.name)(self.dataset_cfg)

    def show_cfgs(self):
        self.logger.info(self.model.cfg)
        self.logger.info(self.run_cfg)
        self.logger.info(self.dataset.cfg)

    def split(self, index_cross):
        self.dataset_cfg.index_cross = index_cross
        self.path = utils.path.get_path(self.model_cfg, self.dataset_cfg, self.run_cfg)

        self.logger = utils.Logger(self.path, self.model.name)
        self.dataset.set_logger(self.logger)

        self.trainset, self.testset = self.dataset.split(index_cross)
        self.train_loader = DataLoader(self.trainset, batch_size=self.run_cfg.batch_size, shuffle=True,
                                       num_workers=0 if platform.system() == 'Windows' else 8, pin_memory=True) \
            if len(self.trainset) > 0 else list()
        self.test_loader = DataLoader(self.testset, batch_size=self.run_cfg.batch_size, shuffle=False,
                                       num_workers=0 if platform.system() == 'Windows' else 8, pin_memory=True) \
            if len(self.testset) > 0 else list()

        self.summary = utils.Summary(self.path, dataset=self.dataset)

        self.model = models.find(self.model_cfg.name)(self.model_cfg, self.dataset.cfg, self.run_cfg,
                                                      summary=self.summary)
        self.start_epoch = self.model.load(self.args.test_epoch)

        self.show_cfgs()

    def train(self, epoch):
        log_step, count = 1, 0
        batch_per_epoch, count_data = len(self.train_loader), len(self.train_loader.dataset)
        epoch_info = {'epoch': epoch, 'batch_per_epoch': batch_per_epoch, 'count_data': count_data}
        for batch_idx, (sample_dict, index) in enumerate(self.train_loader):
            epoch_info['batch_idx'] = batch_idx
            self.summary.update_epochinfo(epoch_info)
            loss_dict = self.model.train(epoch_info, sample_dict)
            count += len(list(sample_dict.values())[0])
            if batch_idx % log_step == 0:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                loss = list()
                if loss_dict:
                    for name, value in loss_dict.items():
                        msg += ' ' + name + ': {:.6f}'
                        loss.append(value.item())
                self.logger.info(msg.format(epoch, count, count_data, 100. * count / count_data, *loss))
        if epoch % self.run_cfg.save_step == 0:
            self.model.save(epoch)

    def test(self, epoch):
        torch.cuda.empty_cache()
        predict = dict()
        log_step, count = 1, 0
        with torch.no_grad():
            for batch_idx, (sample_dict, index) in enumerate(self.test_loader):
                output_dict = self.model.test(batch_idx, sample_dict)
                count += len(list(sample_dict.values())[0])
                if batch_idx % log_step == 0:
                    self.logger.info('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                        epoch, count, len(self.test_loader.dataset), 100. * count / len(self.test_loader.dataset)))
                if len(predict) != len(output_dict):
                    for name, value in output_dict.items():
                        data_type = name.split('_')[-1]
                        data_cfg = getattr(self.dataset.cfg, data_type, None)
                        if data_cfg is not None:
                            predict_shape = (data_cfg.time, data_cfg.width, data_cfg.height) \
                                if data_cfg.elements > 1 else [1]
                            predict[name] = torch.zeros(self.testset.raw_count, *predict_shape)
                        else:
                            predict[name] = torch.Tensor().to(value.device)
                for name, value in output_dict.items():
                    data_type = name.split('_')[-1]
                    data_cfg = getattr(self.dataset.cfg, data_type, None)
                    if data_cfg is not None:
                        for i in range(len(value)):
                            slice_recover = self.testset.recover(index[i])[data_type]
                            predict[name][slice_recover] = value[i]
                    else:
                        predict[name] = torch.cat((predict[name], value.float()
                            if value.shape else torch.tensor([value], dtype=torch.float).to(value.device)))

        for name, value in predict.items():
            predict[name] = np.array(value.cpu())
            if self.dataset_cfg.norm and self.dataset.need_norm(value.shape):
                data_type = name.split('_')[-1]
                data_cfg = getattr(self.dataset.cfg, data_type, None)
                if data_cfg is not None:
                    other = dict()
                    predict[name] = self.dataset.renorm(predict[name], data_type, **other)

        if epoch % 1 == 0:
            predict_file = os.path.join(self.path, self.model.name + '_' + str(epoch)
                                        + configs.env.paths.predict_file)
            if predict:
                scipy.io.savemat(predict_file, predict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Template')
    parser.add_argument('--model_config_path', type=str, required=True, metavar='/path/to/model/config.json',
                        help='Path to model config .json file')
    parser.add_argument('--run_config_path', type=str, required=True, metavar='/path/to/run/config.json',
                        help='Path to run config .json file')
    parser.add_argument('--dataset_config_path', type=str, required=True, metavar='/path/to/dataset/config.json',
                        help='Path to dataset config .json file')
    parser.add_argument('--test_epoch', type=int, metavar='epoch want to test', help='epoch want to test')
    parser.add_argument('--ci', action='store_false' if configs.env.ci.run else 'store_true',
                        default=configs.env.ci.run, help='running CI')
    args = parser.parse_args()
    print(args)

    main = Main(args)
    for index_cross in range(min(main.dataset.cfg.cross_folder, 1), main.dataset.cfg.cross_folder + 1):
        main.split(index_cross)
        if args.test_epoch is None:
            for epoch in range(main.start_epoch + 1, main.run_cfg.epochs + 1):
                main.train(epoch)
                if epoch % main.run_cfg.save_step == 0:
                    main.test(epoch)
        else:
            main.test(main.start_epoch)
