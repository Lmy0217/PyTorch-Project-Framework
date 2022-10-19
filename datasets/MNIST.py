import numpy as np
from torchvision import datasets as Datasets

import configs
import datasets

__all__ = ['MNIST']


class MNIST(datasets.BaseDataset):

    @staticmethod
    def more(cfg):
        cfg.path = configs.env.getdir(cfg.path)
        cfg.source.elements = cfg.source.width * cfg.source.height * cfg.source.time
        cfg.cross_folder = 0
        cfg.index_cross = 0
        return cfg

    def load(self):
        train = Datasets.MNIST(self.cfg.path, train=True, download=True)
        test = Datasets.MNIST(self.cfg.path, train=False, download=True)

        source_train = train.data.numpy().astype(np.float32)[:, np.newaxis, :, :]
        source_test = test.data.numpy().astype(np.float32)[:, np.newaxis, :, :]
        target_train = train.targets.numpy()[:, np.newaxis]
        target_test = test.targets.numpy()[:, np.newaxis]

        data = {
            'source_train': source_train,
            'target_train': target_train,
            'source_test': source_test,
            'target_test': target_test,
        }
        self.count_train, self.count_test = len(train), len(test)
        data_count = self.count_train + self.count_test

        return data, data_count

    def __getitem__(self, index):
        if index < self.count_train:
            source, target = self.data['source_train'][index], self.data['target_train'][index]
        else:
            index = index - self.count_train
            source, target = self.data['source_test'][index], self.data['target_test'][index]
        return {'source': source, 'target': target}, index

    def split(self, index_cross=None):
        self._reset_norm(split_items=[['source_train', 'target_train'], ['source_test', 'target_test']])
        trainset = datasets.BaseSplit(self, [[0, self.count_train]])
        testset = datasets.BaseSplit(self, [[self.count_train, self.cfg.data_count]])
        return trainset, testset


if __name__ == "__main__":
    datasets.BaseTest(MNIST).run()
