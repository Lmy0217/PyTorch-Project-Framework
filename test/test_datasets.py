import unittest
import os
import numpy as np
import json

from configs import BaseConfig, env
from datasets import BaseDataset, SampleDataset
from datasets.functional import norm


class TestBaseDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(env.getdir(env.paths.test_folder),
                                os.path.splitext(os.path.basename(__file__))[0], cls.__name__)
        if not os.path.exists(cls.path):
            os.makedirs(cls.path)

        class SimpleDataset(BaseDataset):

            def _load(self):
                return dict(test=np.transpose(np.tile(np.arange(env.ci.batchsize), (6, 1))),
                            t2=np.tile(np.arange(6), (6, 1))), env.ci.batchsize

            def _recover(self, index):
                return [index], dict(test=[0, self.cfg.test.elements], t2=[0, self.cfg.t2.elements])

        json_path = os.path.join(cls.path, 'test_configs.json')
        with open(json_path, 'w') as f:
            f.write(json.dumps(dict(paths=dict(test='_*??_??', file_count=[2, 10]), cross_folder=2,
                                    test=dict(lower=0, upper=5, elements=6), t2=dict(lower=0, upper=10, elements=6))))

        cls.dataset = SimpleDataset(BaseConfig(json_path), t=1)

    @classmethod
    def tearDownClass(cls):
        pass

    def testInit(self):
        self.assertEqual(self.dataset.name, 'test_configs')
        self.assertEqual(self.dataset.cfg.test.elements, 6)
        self.assertEqual(self.dataset.cfg.out.elements, 1)
        self.assertEqual(self.dataset.data['test'].shape, (env.ci.batchsize, 6))
        self.assertEqual(self.dataset.cfg.data_count, env.ci.batchsize, 6)
        self.assertEqual(self.dataset.t, 1)

    def test_path(self):
        self.assertEqual(self.dataset._path([1, 10]), dict(test='_*01_10'))
        self.assertEqual(self.dataset._path([2, 1]), dict(test='_*02_01'))

    def test_neednorm(self):
        self.assertTrue(self.dataset.need_norm((env.ci.batchsize, 6)))
        self.assertFalse(self.dataset.need_norm((env.ci.batchsize, 1, 1)))

    def test_meanstd(self):
        meanstd = self.dataset.meanstd(self.dataset.data['test'])
        self.assertTrue((meanstd[0] == 1.5).all())
        self.assertTrue((meanstd[1] == np.std([0, 1, 2, 3])).all())

    def test_ms(self):
        ms = self.dataset._ms(norm_set=[0, 1, 2])
        self.assertTrue((ms['test'][0] == 1).all())
        self.assertTrue((ms['test'][1] == np.std([0, 1, 2])).all())
        self.assertTrue((ms['t2'][0] == np.arange(6)).all())
        self.assertTrue((ms['t2'][1] == 0).all())

        ms = self.dataset._ms(split_items=[['test'], ['t2']])
        self.assertTrue((ms['test'][0] == 1.5).all())
        self.assertTrue((ms['test'][1] == np.std([0, 1, 2, 3])).all())
        self.assertTrue((ms['t2'][0] == 1.5).all())
        self.assertTrue((ms['t2'][1] == np.std([0, 1, 2, 3])).all())

        ms = self.dataset._ms(norm_set=[0, 1, 2], split_items=[['test'], ['t2']])
        self.assertTrue((ms['test'][0] == 1).all())
        self.assertTrue((ms['test'][1] == np.std([0, 1, 2])).all())
        self.assertTrue((ms['t2'][0] == 1).all())
        self.assertTrue((ms['t2'][1] == np.std([0, 1, 2])).all())

    def test_reset_norm(self):
        raw_test = np.transpose(np.tile(np.arange(env.ci.batchsize), (6, 1)))
        raw_t2 = np.tile(np.arange(6), (6, 1))

        self.assertEqual(self.dataset.cfg.norm, 'threshold')
        self.dataset._reset_norm()
        self.assertIsNone(self.dataset.cfg.norm)
        self.dataset.cfg.norm = 'threshold'

        self.dataset._reset_norm(norm_range=[[0, 1], [2, 4]])
        self.assertEqual(self.dataset.split_set, [0, 2, 3])
        self.assertIsNone(self.dataset.split_item)
        self.assertTrue((self.dataset.ms[-1]['test'][0] == 5 / 3).all())
        self.assertTrue((self.dataset.ms[-1]['test'][1] == np.std([0, 2, 3])).all())
        self.assertTrue((self.dataset.data['test'] == norm.threshold_norm(raw_test, 0, 5)).all())
        self.assertTrue((self.dataset.ms[-1]['t2'][0] == np.arange(6)).all())
        self.assertTrue((self.dataset.ms[-1]['t2'][1] == 0).all())
        self.assertTrue((self.dataset.data['t2'] == norm.threshold_norm(raw_t2, 0, 10)).all())

        self.dataset._reset_norm(split_items=[['test'], ['t2']])
        self.assertIsNone(self.dataset.split_set)
        self.assertEqual(self.dataset.split_item, [['test'], ['t2']])
        self.assertTrue((self.dataset.ms[-1]['test'][0] == 1.5).all())
        self.assertTrue((self.dataset.ms[-1]['test'][1] == np.std([0, 1, 2, 3])).all())
        self.assertTrue((self.dataset.data['test'] == norm.threshold_norm(raw_test, 0, 5)).all())
        self.assertTrue((self.dataset.ms[-1]['t2'][0] == 1.5).all())
        self.assertTrue((self.dataset.ms[-1]['t2'][1] == np.std([0, 1, 2, 3])).all())
        self.assertTrue((self.dataset.data['t2'] == norm.threshold_norm(raw_t2, 0, 10)).all())

        self.dataset._reset_norm(norm_range=[[0, 1], [2, 4]], split_items=[['test'], ['t2']])
        self.assertEqual(self.dataset.split_set, [0, 2, 3])
        self.assertEqual(self.dataset.split_item, [['test'], ['t2']])
        self.assertTrue((self.dataset.ms[-1]['test'][0] == 5 / 3).all())
        self.assertTrue((self.dataset.ms[-1]['test'][1] == np.std([0, 2, 3])).all())
        self.assertTrue((self.dataset.data['test'] == norm.threshold_norm(raw_test, 0, 5)).all())
        self.assertTrue((self.dataset.ms[-1]['t2'][0] == 5 / 3).all())
        self.assertTrue((self.dataset.ms[-1]['t2'][1] == np.std([0, 2, 3])).all())
        self.assertTrue((self.dataset.data['t2'] == norm.threshold_norm(raw_t2, 0, 10)).all())

        self.dataset._renorm()
        delattr(self.dataset, 'ms')
        delattr(self.dataset, 'split_set')
        delattr(self.dataset, 'split_item')
        self.dataset.cfg.norm = env.dataset.norm

    def test_norm(self):
        data = self.dataset.data['t2']
        self.dataset.ms = [self.dataset._ms(split_items=[['test'], ['t2']])]
        mean = self.dataset.ms[-1]['t2'][0]
        std = self.dataset.ms[-1]['t2'][1]

        self.dataset.cfg.norm = 'standard'
        self.assertTrue((self.dataset.norm(data, 't2') == norm.standard_norm(data, mean, std)).all())
        self.dataset.cfg.norm = 'triple'
        self.assertTrue((self.dataset.norm(data, 't2') == norm.triple_norm(data, mean, std)).all())
        self.dataset.cfg.norm = 'threshold'
        self.assertTrue((self.dataset.norm(data, 't2') == norm.threshold_norm(data, 0, 10)).all())

        delattr(self.dataset, 'ms')
        self.dataset.cfg.norm = env.dataset.norm

    def test_renorm(self):
        data = self.dataset.data['t2']
        self.dataset.ms = [self.dataset._ms(split_items=[['test'], ['t2']])]
        mean = self.dataset.ms[-1]['t2'][0]
        std = self.dataset.ms[-1]['t2'][1]

        self.dataset.cfg.norm = 'standard'
        self.assertTrue((self.dataset.renorm(data, 't2') == norm.standard_renorm(data, mean, std)).all())
        self.dataset.cfg.norm = 'triple'
        self.assertTrue((self.dataset.renorm(data, 't2') == norm.triple_renorm(data, mean, std)).all())
        self.dataset.cfg.norm = 'threshold'
        self.assertTrue((self.dataset.renorm(data, 't2') == norm.threshold_renorm(data, 0, 10)).all())

        delattr(self.dataset, 'ms')
        self.dataset.cfg.norm = env.dataset.norm

    def test_get_args(self):
        data = self.dataset.data['t2']
        self.dataset.ms = [self.dataset._ms(split_items=[['test'], ['t2']])]

        for fn_name in dir(norm):
            if fn_name.endswith('_norm') or fn_name.endswith('_renorm'):
                fn = getattr(norm, fn_name)
                self.assertEqual(tuple(self.dataset._get_args(fn, data, 't2').keys()), fn.__code__.co_varnames)

        delattr(self.dataset, 'ms')
        self.dataset.cfg.norm = env.dataset.norm

    def test_recover_(self):
        indexes, index_dict = self.dataset._recover(1)
        self.assertEqual(indexes, [1])
        self.assertEqual(index_dict, dict(test=[0, 6], t2=[0, 6]))

    def test_recover(self):
        index_dict = self.dataset.recover(1)
        self.assertEqual(index_dict, dict(test=(1, slice(0, 6)), t2=(1, slice(0, 6))))

    def test_getitem(self):
        sample_dict, index = self.dataset[1]
        self.assertTrue((sample_dict['test'] == np.array([1, 1, 1, 1, 1, 1])).all())
        self.assertTrue((sample_dict['t2'] == np.array([0, 1, 2, 3, 4, 5])).all())
        self.assertEqual(index, 1)

    def test_len(self):
        self.assertEqual(len(self.dataset), env.ci.batchsize)

    def test_cross(self):
        old_cross_folder = self.dataset.cfg.cross_folder

        self.dataset.cfg.cross_folder = 0
        self.assertEqual(self.dataset._cross(0), ([[0, 4]], [], [[0, 4]], [0, 0, 0, 0]))
        self.dataset.cfg.cross_folder = 1
        self.assertEqual(self.dataset._cross(1), ([[0, 0], [4, 4]], [[0, 4]], [[0, 0], [4, 4]], [0, 4, 0, 4]))
        self.dataset.cfg.cross_folder = 2
        self.assertEqual(self.dataset._cross(1), ([[0, 0], [2, 4]], [[0, 2]], [[0, 0], [2, 4]], [0, 2, 0, 2]))
        self.assertEqual(self.dataset._cross(1, data_count=2, elements_per_data=2),
                         ([[0, 0], [2, 4]], [[0, 2]], [[0, 0], [2, 4]], [0, 2, 0, 2]))
        self.assertEqual(self.dataset._cross(2), ([[0, 2], [4, 4]], [[2, 4]], [[0, 2], [4, 4]], [2, 2, 2, 2]))
        self.assertEqual(self.dataset._cross(2, data_count=2, elements_per_data=2),
                         ([[0, 2], [4, 4]], [[2, 4]], [[0, 2], [4, 4]], [2, 2, 2, 2]))

        self.dataset.cfg.cross_folder = old_cross_folder

    def test_split(self):
        trainset, testset = self.dataset.split(1)
        self.assertEqual(len(trainset), 2)
        self.assertEqual(len(testset), 2)
        self.assertTrue((trainset[1][0]['test'] == self.dataset[3][0]['test']).all())
        self.assertTrue((testset[1][0]['test'] == self.dataset[1][0]['test']).all())
        self.assertEqual(trainset.recover(0), dict(test=(0, slice(0, 6)), t2=(0, slice(0, 6))))
        self.assertEqual(testset.recover(0), dict(test=(0, slice(0, 6)), t2=(0, slice(0, 6))))


class TestSampleDataset(unittest.TestCase):

    def test_SampleDataset(self):
        sample = dict(x=np.array([[1, 2], [3, 4]]), y=np.array([[4, 2], [3, 1]]))
        sample_dataset = SampleDataset(sample)
        return_sample, index = sample_dataset[0]
        self.assertTrue((return_sample['x'] == np.array([[1, 2]])).all())
        self.assertTrue((return_sample['y'] == np.array([[4, 2]])).all())
        self.assertEqual(index, 0)
        self.assertEqual(len(sample_dataset), 2)


class TestFunctional_norm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = np.array([[1, 1], [2, 2]])
        cls.mean = np.array([[2, 2], [2, 2]])
        cls.std = np.array([[0.5, 0.5], [0.5, 0.5]])
        cls.lower = 1
        cls.upper = 2

    @classmethod
    def tearDownClass(cls):
        pass

    def test_standard_norm(self):
        data_norm = norm.standard_norm(self.data, self.mean, self.std)
        self.assertTrue((data_norm == np.array([[-2, -2], [0, 0]])).all())

    def test_standard_renorm(self):
        data_norm = norm.standard_renorm(self.data, self.mean, self.std)
        self.assertTrue((data_norm == np.array([[2.5, 2.5], [3, 3]])).all())

    def test_triple_norm(self):
        data_norm = norm.triple_norm(self.data, self.mean, self.std)
        self.assertTrue((data_norm == np.array([[-2 / 3, -2 / 3], [0, 0]])).all())

    def test_triple_renorm(self):
        data_norm = norm.triple_renorm(self.data, self.mean, self.std)
        self.assertTrue((data_norm == np.array([[3.5, 3.5], [5, 5]])).all())

    def test_threshold_norm(self):
        data_norm = norm.threshold_norm(self.data, self.lower, self.upper)
        self.assertTrue((data_norm == np.array([[-1, -1], [1, 1]])).all())

    def test_threshold_renorm(self):
        data_norm = norm.threshold_renorm(self.data, self.lower, self.upper)
        self.assertTrue((data_norm == np.array([[2, 2], [2.5, 2.5]])).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
