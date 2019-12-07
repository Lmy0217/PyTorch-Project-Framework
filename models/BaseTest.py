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
                logger = utils.Logger('test', model_name)
                logger.info('Testing model: ' + model_name + ' ...')
                for data_cfg in datasets.allcfgs():
                    if not self.model.check_cfg(data_cfg, model_cfg):
                        continue
                    data_name = os.path.splitext(os.path.split(data_cfg._path)[1])[0]
                    logger.info('\tTesting dataset: ' + data_name + ' ...')
                    data_cfg.index_cross = 1
                    sample_dict = dict()
                    for name, value in vars(data_cfg).items():
                        if name.startswith('source') or name.startswith('target'):
                            kernel = getattr(data_cfg, 'kernel' if name.startswith('source') else 'out_kernel', None)
                            if kernel is not None:
                                sample_shape = (kernel.kT, kernel.kW, kernel.kH)
                                sample_dict[name] = torch.randn(configs.env.ci.batchsize, *sample_shape)
                            else:
                                sample_shape = (value.time, value.width, value.height) \
                                    if hasattr(value, 'time') else [value.elements]
                                sample_dict[name] = torch.randint(value.classes, (configs.env.ci.batchsize, 1)).long() \
                                    if len(sample_shape) == 1 and sample_shape[0] == 1 \
                                    else torch.randn(configs.env.ci.batchsize, *sample_shape)
                            logger.info("\t-- " + name + " size: " + str(sample_dict[name].size()))

                    for run_cfg in configs.Run.all():
                        run_name = os.path.splitext(os.path.split(run_cfg._path)[1])[0]
                        logger.info('\t\tTesting config: ' + run_name + ' ...')
                        model = self.model(model_cfg, data_cfg, run_cfg)

                        params, params_all = dict(), 0
                        for name, value in model.modules().items():
                            params[name] = sum(p.numel() for p in value.parameters() if p.requires_grad)
                            params_all += params[name]
                        logger.info("\t\t-- parameter(s): " + str(params))
                        logger.info("\t\t-- all parameters: " + str(params_all))

                        loss_dict = model.train(0, sample_dict)
                        logger.info("\t\t-- loss(es): " + str(loss_dict))

                        torch.cuda.empty_cache()
                        result_dict = model.test(0, sample_dict)
                        for name, value in result_dict.items():
                            result_dict[name] = value.shape
                        logger.info("\t\t-- result(s) size: " + str(result_dict))
                        logger.info("\t\t-- save folder: " + str(model.getpath()))

                        save_folder = os.path.join("test", model_name, data_name + '-' + run_name)
                        model.save(epoch=0, path=save_folder)
                        model.load(path=save_folder)
                        logger.info('\t\tTesting config: ' + run_name + ' completed.')
                    logger.info('\tTesting dataset: ' + data_name + ' completed.')
                logger.info('Testing model: ' + model_name + ' completed.')
