import os
import shutil

import torch
from torch.utils.data import DataLoader

import configs
import datasets
import models
import utils

__all__ = ['BaseTest']


class BaseTest(object):

    def __init__(self, model):
        self.ci = configs.env.ci.run
        configs.env.ci.run = True
        self.bs = configs.env.ci.batchsize
        configs.env.ci.batchsize = 2
        self.model = model

    def __del__(self):
        configs.env.ci.run = self.ci
        configs.env.ci.batchsize = self.bs

    def run(self, rm_save_folder=True, one_dataset=False):
        for model_cfg in models.functional.common.allcfgs():
            if hasattr(model_cfg, 'name') and model_cfg.name == self.model.__name__:
                model_name = os.path.splitext(os.path.split(model_cfg._path)[1])[0]
                logger = utils.Logger(os.path.join(os.path.dirname(__file__), 'test', model_name), model_name)
                logger.info('Testing model: ' + model_name + ' ...')

                for data_cfg in datasets.functional.common.allcfgs():
                    if not self.model.check_cfg(data_cfg, model_cfg):
                        # print("\tDataset '" + data_cfg.name + "' not support")
                        continue
                    dataset = datasets.functional.common.find(data_cfg.name)(data_cfg)
                    dataset.set_logger(logger)
                    dataset.set_summary(None)
                    logger.info('\tTesting dataset: ' + dataset.name + ' ...')

                    data_cfg.index_cross = 1
                    splitsets = dataset.split(1)
                    trainset, testset = splitsets[0], splitsets[-1]
                    test_batchsize = configs.env.ci.batchsize
                    sample_loader = DataLoader(trainset, batch_size=configs.env.ci.batchsize, pin_memory=trainset.dataset.cfg.pin_memory)
                    test_sample_loader = DataLoader(testset, batch_size=test_batchsize, pin_memory=testset.dataset.cfg.pin_memory)

                    for run_cfg in configs.Run.all():
                        if run_cfg.name not in ['sp1']:
                            continue
                        run_name = os.path.splitext(os.path.split(run_cfg._path)[1])[0]
                        logger.info('\t\tTesting config: ' + run_name + ' ...')

                        save_folder = os.path.join(os.path.dirname(__file__), 'test', model_name, dataset.name + '-' + run_name)
                        summary = utils.Summary(save_folder, dataset=dataset)
                        dataset.set_summary(summary)
                        main_msg = dict(ci='ci')
                        main_msg.update(dict(index_cross=data_cfg.index_cross, while_idx=1, while_flag=True))

                        model = self.model(model_cfg, data_cfg, run_cfg, summary=summary, main_msg=main_msg)

                        params, params_all = dict(), 0
                        for name, value in model.modules().items():
                            params[name] = sum(p.numel() for p in value.parameters() if p.requires_grad)
                            params_all += params[name]
                        logger.info("\t\t-- parameter(s): " + str(params))
                        logger.info("\t\t-- all parameters: " + str(params_all))

                        while main_msg['while_flag']:
                            model.process_pre_hook()
                            model.main_msg = main_msg
                            sample_loader = model.train_loader_hook(sample_loader)
                            epoch_info = {'epoch': 1, 'batch_idx': 0, 'index': torch.arange(configs.env.ci.batchsize),
                                          'batch_count': configs.env.ci.batchsize, 'batch_per_epoch': 1,
                                          'count_data': configs.env.ci.batchsize}
                            summary.update_epochinfo(epoch_info)
                            model.train_epoch_pre_hook(epoch_info, sample_loader)
                            for batch_idx, (sample_dict, index) in enumerate(sample_loader):
                                loss_all = dict()
                                loss_dict = model.train_process(epoch_info, sample_dict)
                                loss_dict.update(dict(_count=[configs.env.ci.batchsize]))
                                utils.common.merge_dict(loss_all, loss_dict)
                                model.train_epoch_hook(epoch_info, sample_loader)
                                loss_all = model.train_return_hook(epoch_info, loss_all)
                                logger.info("\t\t-- loss(es) " + str(main_msg['while_idx']) + ": " + str(loss_all))
                                break

                            model.main_msg.update(dict(test_idx=1, test_flag=True, only_test=False))
                            while model.main_msg['test_flag']:
                                torch.cuda.empty_cache()
                                with torch.no_grad():
                                    test_sample_loader = model.test_loader_hook(test_sample_loader)
                                    epoch_info.update(dict(index=torch.arange(test_batchsize), batch_count=test_batchsize, count_data=test_batchsize, log_text='Test'))
                                    model.test_epoch_pre_hook(epoch_info, test_sample_loader)
                                    result_dict = {}
                                    for batch_idx, (test_sample_dict, index) in enumerate(test_sample_loader):
                                        result_dict = model.test_process(epoch_info, test_sample_dict)
                                        break
                                    model.test_epoch_hook(epoch_info, test_sample_loader)
                                    result_dict = model.test_return_hook(
                                        epoch_info, {
                                            k: v.detach().cpu().numpy() for k, v in result_dict.items()
                                            if isinstance(v, torch.Tensor)
                                        } if isinstance(result_dict, dict) else result_dict)
                                    add_data_msgs, msgs = None, None
                                    if isinstance(result_dict, tuple):
                                        if len(result_dict) == 2:
                                            result_dict, add_data_msgs = result_dict
                                        elif len(result_dict) == 3:
                                            result_dict, add_data_msgs, msgs = result_dict
                                    for name, value in result_dict.items():
                                        result_dict[name] = value.shape
                                    logger.info("\t\t-- result(s) " + str(model.main_msg['test_idx']) + " size: " + str(result_dict))
                                    if msgs is not None:
                                        logger.info("\t\t-- msg(s): " + str(msgs))
                                model.process_test_msg_hook(model.main_msg)
                            model.process_hook()
                            model.process_msg_hook(main_msg)

                        logger.info("\t\t-- save folder: " + str(utils.path.get_path(model_cfg, data_cfg, run_cfg)))

                        model.save(epoch=1, path=save_folder)
                        model.load(path=save_folder)
                        if rm_save_folder:
                            shutil.rmtree(save_folder)
                        logger.info('\t\tTesting config: ' + run_name + ' completed.')
                        break

                    logger.info('\tTesting dataset: ' + dataset.name + ' completed.')
                    if one_dataset:
                        break

                logger.info('Testing model: ' + model_name + ' completed.')
