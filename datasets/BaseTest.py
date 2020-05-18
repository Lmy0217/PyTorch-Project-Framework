import datasets
import utils
import os
import numpy as np


__all__ = ['BaseTest']


class BaseTest(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def run(self):
        for dataset_cfg in datasets.functional.common.allcfgs():
            if hasattr(dataset_cfg, 'name') and dataset_cfg.name == self.dataset.__name__:
                dataset_name = os.path.splitext(os.path.split(dataset_cfg._path)[1])[0]
                save_folder = os.path.join(os.path.dirname(__file__), 'test', dataset_name)
                logger = utils.Logger(save_folder, dataset_name)
                logger.info('Testing dataset: ' + dataset_name + ' ...')
                dataset = self.dataset(dataset_cfg)
                dataset.set_logger(logger)
                summary = utils.Summary(save_folder, dataset=dataset)
                dataset.set_summary(summary)
                trainset, testset = dataset.split(
                    index_cross=min(dataset.cfg.cross_folder, 1) if hasattr(dataset.cfg, 'cross_folder') else None)

                for splitset, set_name in zip([trainset, testset], ['Trainset', 'Testset']):
                    logger.info("-- " + set_name + " size: " + str(len(splitset)))
                    log_step = max(int(
                        np.power(10, np.floor(np.log10(len(splitset) / 100)))), 1) if len(splitset) > 0 else 0
                    for i in range(len(splitset)):
                        sample_info = "  -- The " + str(i + 1) + "-th sample:"
                        sample_dict, index = splitset[i]
                        for name, value in sample_dict.items():
                            if (hasattr(value, 'ndim') and value.ndim > 1) or value.shape[0] > 1:
                                sample_info += " " + name + " size: " + str(value.shape)
                            else:
                                sample_info += " " + name + " : " + str(value)
                        if (i + 1) % log_step == 0 or i == len(splitset) - 1:
                            logger.info(sample_info)
                logger.info('Testing dataset: ' + dataset_name + ' completed.')
