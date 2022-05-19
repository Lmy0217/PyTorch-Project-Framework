import os

import configs
import utils

__all__ = ['BaseTest']


class BaseTest(object):

    def __init__(self, path):
        self.path = path if isinstance(path, list) else [path]

    def run(self):
        for p in self.path:
            name = os.path.splitext(os.path.split(p)[1])[0]
            logger = utils.Logger(os.path.join(os.path.dirname(__file__), 'test'), name)
            if isinstance(p, str) or isinstance(p, dict):
                logger.info('Testing ' + str(p) + ' ...')
                logger.info(configs.BaseConfig(p))
                logger.info('Testing ' + str(p) + ' completed.')
            else:
                raise TypeError('got {} but need dict or str!'.format(type(p)))
