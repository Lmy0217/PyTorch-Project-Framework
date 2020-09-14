import configs
import datasets
import numpy as np
from torchvision import datasets as Datasets


__all__ = ['MNIST']


class MNIST(datasets.BaseDataset):

    def __init__(self, cfg, **kwargs):
        super(MNIST, self).__init__(cfg, **kwargs)

    @staticmethod
    def more(cfg):
        cfg.path = configs.env.getdir(cfg.path)
        cfg.source.elements = cfg.source.width * cfg.source.height * cfg.source.time
        out = dict()
        out['elements'] = 1
        cfg.out = configs.BaseConfig(out)
        cfg.cross_folder = 0
        cfg.index_cross = 0
        return cfg

    def load(self):
        train = Datasets.MNIST(self.cfg.path, train=True, download=True)
        test = Datasets.MNIST(self.cfg.path, train=False, download=True)

        source_train, source_test = train.data.numpy()[:, np.newaxis, :, :], test.data.numpy()[:, np.newaxis, :, :]
        source_train, source_test = source_train.astype(np.float32), source_test.astype(np.float32)
        target_train, target_test = train.targets.numpy()[:, np.newaxis], test.targets.numpy()[:, np.newaxis]
        count_train, count_test = len(train), len(test)
        count = count_train + count_test

        return {'source_train': source_train, 'target_train': target_train,
                'source_test': source_test, 'target_test': target_test}, \
               {'count_train': count_train, 'count_test': count_test, 'count': count}

    def __getitem__(self, index):
        if index < self.cfg.data_count['count_train']:
            source, target = self.data['source_train'][index], self.data['target_train'][index]
        else:
            index = index - self.cfg.data_count['count_train']
            source, target = self.data['source_test'][index], self.data['target_test'][index]
        return {'source': source, 'target': target}, index

    def split(self, index_cross=None):
        self._reset_norm(split_items=[['source_train', 'target_train'], ['source_test', 'target_test']])

        trainset = datasets.BaseSplit(self, [[0, self.cfg.data_count['count_train']]])
        testset = datasets.BaseSplit(self, [[self.cfg.data_count['count_train'], self.cfg.data_count['count']]])

        return trainset, testset


if __name__ == "__main__":
    datasets.BaseTest(MNIST).run()
