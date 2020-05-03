import unittest
import os
import torch
import numpy as np

from configs import env
from utils.common import deepcopy, merge_dict
from utils.path import get_filename, get_path
from utils.image import _2dto3d, sobel3d, laplace3d
from utils.medical import cbf


class TestCommon(unittest.TestCase):

    def test_deepcopy(self):
        class s: pass

        a = s()
        a.b = 1
        a.c = [1, 2]

        t1 = deepcopy(a)
        t1.c[0] = 3
        self.assertEqual(a.c, [1, 2])
        self.assertEqual(t1.c, [3, 2])

        t2 = deepcopy(a, ['c'])
        t2.c[0] = 5
        self.assertEqual(a.c, [5, 2])
        self.assertEqual(t2.c, [5, 2])

    def test_merge_dict(self):
        a = dict()

        merge_dict(a, dict(c=torch.tensor(1)))
        self.assertEqual(a, dict(c=[1]))

        merge_dict(a, dict(c=1.0))
        self.assertEqual(a, dict(c=[1, 1.0]))


class TestPath(unittest.TestCase):

    def test_get_filename(self):
        file_path = "/user/file"
        self.assertEqual(get_filename(file_path), 'file')

    def test_get_path(self):
        class p:
            _path: str

        m_cfg = p()
        m_cfg._path = 'm'
        d_cfg = p()
        d_cfg._path = 'd'
        d_cfg.index_cross = 1
        r_cfg = p()
        r_cfg._path = 'r'

        self.assertEqual(get_path(m_cfg, d_cfg, r_cfg),
                         os.path.join(env.getdir(env.paths.save_folder), 'm-r-d-1'))


class TestLogger(unittest.TestCase):
    # TODO test logger
    def testInit(self):
        pass


class TestSummary(unittest.TestCase):
    # TODO test summary
    def testInit(self):
        pass


class TestImage(unittest.TestCase):

    def test_2dto3d(self):
        a = np.random.rand(1, 3, 6, 6)
        b = _2dto3d(a)
        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[2:], a.shape[1:])

    def test_sobel3d(self):
        # TODO more test
        a = torch.randn(1, 3, 6, 6)
        b = np.random.rand(2, 3, 6, 6)
        sobel = sobel3d(a)
        self.assertEqual(a.shape, sobel.shape)
        sobel_np = sobel3d(b)
        self.assertEqual(b.shape, sobel_np.shape)

    def test_laplace3d(self):
        # TODO more test
        a = torch.randn(1, 3, 6, 6)
        b = np.random.rand(2, 3, 6, 6)
        laplace = laplace3d(a)
        self.assertEqual(a.shape, laplace.shape)
        laplace_np = laplace3d(b)
        self.assertEqual(b.shape, laplace_np.shape)


class TestMedical(unittest.TestCase):

    def test_cbf(self):
        # TODO more test
        a = torch.randn((2, 3, 6, 6), requires_grad=True)
        b = torch.randn((2, 3, 6, 6))
        c = cbf(a, b)
        self.assertEqual(c.shape, a.shape)
        self.assertTrue(c.requires_grad)

        a = np.random.randn(2, 3, 6, 6)
        b = np.random.randn(2, 3, 6, 6)
        c = cbf(a, b)
        self.assertEqual(c.shape, a.shape)


if __name__ == '__main__':
    unittest.main(verbosity=2)
