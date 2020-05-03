from torch.utils.data import DataLoader
import torch
import os
import configs
import datasets
import models
import utils


class BaseTest(object):

    def __init__(self, model):
        self.model = model

    def run(self):
        for model_cfg in models.allcfgs():
            if hasattr(model_cfg, 'name') and model_cfg.name == self.model.__name__:
                model_name = os.path.splitext(os.path.split(model_cfg._path)[1])[0]
                logger = utils.Logger(os.path.join(os.path.dirname(__file__), 'test'), model_name)
                logger.info('Testing model: ' + model_name + ' ...')
                for data_cfg in datasets.allcfgs():
                    if not self.model.check_cfg(data_cfg, model_cfg):
                        # print("\tDataset '" + data_cfg.name + "' not support")
                        continue
                    data_name = os.path.splitext(os.path.split(data_cfg._path)[1])[0]
                    logger.info('\tTesting dataset: ' + data_name + ' ...')
                    data_cfg.index_cross = 1
                    sample_dict, test_sample_dict = dict(), dict()
                    for name, value in vars(data_cfg).items():
                        if name.startswith('source') or name.startswith('target'):
                            kernel = getattr(data_cfg, 'kernel' if name.startswith('source') else 'out_kernel', None)
                            # TODO support 4-dim
                            if kernel is not None:
                                sample_shape = (kernel.kT, kernel.kW, kernel.kH)
                                sample_dict[name] = torch.randn(configs.env.ci.batchsize, *sample_shape)
                            else:
                                sample_shape = (value.time, value.width, value.height) \
                                    if hasattr(value, 'time') else [value.elements]
                                # TODO re-ID target class in [-1, 1] not in [0, 1]
                                sample_dict[name] = torch.randint(value.classes, (configs.env.ci.batchsize, 1)).long() \
                                    if len(sample_shape) == 1 and sample_shape[0] == 1 \
                                    else torch.randn(configs.env.ci.batchsize, *sample_shape)
                            logger.info("\t-- " + name + " size: " + str(sample_dict[name].size()))
                        elif name.startswith('test_source') or name.startswith('test_target'):
                            test_kernel = getattr(
                                data_cfg, 'test_kernel' if name.startswith('test_source') else 'test_out_kernel', None)
                            if test_kernel is not None:
                                test_sample_shape = (test_kernel.kT, test_kernel.kW, test_kernel.kH)
                                test_sample_dict[name[5:]] = torch.randn(configs.env.ci.batchsize, *test_sample_shape)
                            else:
                                test_sample_shape = (value.time, value.width, value.height) \
                                    if hasattr(value, 'time') else [value.elements]
                                test_sample_dict[name[5:]] = \
                                    torch.randint(value.classes, (configs.env.ci.batchsize, 1)).long() \
                                        if len(test_sample_shape) == 1 and test_sample_shape[0] == 1 \
                                        else torch.randn(configs.env.ci.batchsize, *test_sample_shape)
                            logger.info("\t-- " + name + " size: " + str(test_sample_dict[name[5:]].size()))
                    for name, value in sample_dict.items():
                        if name not in test_sample_dict.keys():
                            test_sample_dict[name] = value
                    sample_loader = DataLoader(datasets.SampleDataset(sample_dict), pin_memory=True)
                    test_sample_loader = DataLoader(datasets.SampleDataset(test_sample_dict), pin_memory=True)

                    for run_cfg in configs.Run.all():
                        run_name = os.path.splitext(os.path.split(run_cfg._path)[1])[0]
                        logger.info('\t\tTesting config: ' + run_name + ' ...')

                        save_folder = os.path.join(os.path.dirname(__file__), 'test',
                                                   model_name, data_name + '-' + run_name)
                        summary = utils.Summary(save_folder)
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
                            loss_all = dict()
                            loss_dict = model.train_process(epoch_info, sample_dict)
                            loss_dict.update(dict(_count=[configs.env.ci.batchsize]))
                            utils.merge_dict(loss_all, loss_dict)
                            model.train_epoch_hook(epoch_info, sample_loader)
                            loss_all = model.train_return_hook(epoch_info, loss_all)
                            logger.info("\t\t-- loss(es) " + str(main_msg['while_idx']) + ": " + str(loss_all))

                            model.main_msg.update(dict(test_idx=1, test_flag=True, only_test=False))
                            while model.main_msg['test_flag']:
                                torch.cuda.empty_cache()
                                test_sample_loader = model.test_loader_hook(test_sample_loader)
                                model.test_epoch_pre_hook(epoch_info, test_sample_loader)
                                result_dict = model.test_process(epoch_info, test_sample_dict)
                                model.test_epoch_hook(epoch_info, test_sample_loader)
                                result_dict = model.test_return_hook(epoch_info, result_dict)
                                add_data_msgs, msgs = None, None
                                if isinstance(result_dict, tuple):
                                    if len(result_dict) == 2:
                                        add_data_msgs = result_dict[1]
                                        result_dict = result_dict[0]
                                    elif len(result_dict) == 3:
                                        msgs = result_dict[2]
                                        add_data_msgs = result_dict[1]
                                        result_dict = result_dict[0]
                                for name, value in result_dict.items():
                                    result_dict[name] = value.shape
                                logger.info("\t\t-- result(s) " + str(model.main_msg['test_idx']) + " size: "
                                            + str(result_dict))
                                if msgs is not None:
                                    logger.info("\t\t-- msg(s): " + str(msgs))
                                model.process_test_msg_hook(model.main_msg)
                            model.process_hook()
                            model.process_msg_hook(main_msg)

                        logger.info("\t\t-- save folder: " + str(utils.path.get_path(model_cfg, data_cfg, run_cfg)))

                        model.save(epoch=1, path=save_folder)
                        model.load(path=save_folder)
                        logger.info('\t\tTesting config: ' + run_name + ' completed.')
                        break
                    logger.info('\tTesting dataset: ' + data_name + ' completed.')
                    # break
                logger.info('Testing model: ' + model_name + ' completed.')
