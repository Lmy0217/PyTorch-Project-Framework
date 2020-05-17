import unittest
import json
import os
import torch

from configs import all, BaseConfig, env, Run


__all__ = ['TestInit', 'TestBaseConfig', 'TestEnv', 'TestRun']


class TestInit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(env.getdir(env.paths.test_folder),
                                os.path.splitext(os.path.basename(__file__))[0], cls.__name__)
        if not os.path.exists(cls.path):
            os.makedirs(cls.path)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_all(self):
        json_path = os.path.join(self.path, 'test_configs.json')
        with open(json_path, 'w') as f:
            f.write(json.dumps(dict(name='a', t=1, _t=2)))
        cfg_a = BaseConfig(json_path)

        cfg_b = all(BaseConfig, self.path)[0]

        self.assertTrue(cfg_a == cfg_b)


class TestBaseConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(env.getdir(env.paths.test_folder),
                                os.path.splitext(os.path.basename(__file__))[0], cls.__name__)
        if not os.path.exists(cls.path):
            os.makedirs(cls.path)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_values(self):
        cfg = BaseConfig(dict(name=3, t=1, _t=2))

        self.assertTrue(cfg._values('t'))
        self.assertFalse(cfg._values('_t'))
        self.assertFalse(cfg._values('name'))

    def test_dict(self):
        cfg = BaseConfig(dict(name=3, t=dict(name='a'), _t=2))

        self.assertTrue(cfg.dict() == dict(t=BaseConfig(dict(name='a'))))
        self.assertTrue(cfg.t.dict() == dict())

    def test_eq(self):
        cfg_a = BaseConfig(dict(name='_t', t=1, _t=2))
        cfg_b = BaseConfig(dict(name='_t', t=1, _t=2, a=3))
        cfg_c = BaseConfig(dict(name='_t', t=1, a=3))
        cfg_d = BaseConfig(dict(name='_a', t=1, _t=2))

        self.assertFalse(cfg_a == cfg_b)
        self.assertFalse(cfg_a == cfg_c)
        self.assertTrue(cfg_b == cfg_c)
        self.assertTrue(cfg_a == cfg_d)

        cfg_e = BaseConfig(dict(name='_t', t=BaseConfig(dict(name='a')), _t=2))
        cfg_f = BaseConfig(dict(name='_t', t=BaseConfig(dict(name='_a')), _t=2))

        self.assertTrue(cfg_e == cfg_f)

    def test_repr(self):
        cfg_a = BaseConfig(dict(name='n', a=1, b=dict(name='t', _t=1), _t=2))
        cfg_a_repr = "BaseConfig (n): {\n  a: 1\n  b: {\n  }\n}\n"

        self.assertEqual(str(cfg_a), cfg_a_repr)
        self.assertEqual(cfg_a.b._space, 2)

    def test_load(self):
        cfg_dict = dict(name='n', a=1, b=dict(name='t', _t=1), _t=2)
        cfg_a = BaseConfig(cfg_dict)

        cfg_b = BaseConfig(dict())
        cfg_b.a = 1
        cfg_b.b = BaseConfig(dict())

        self.assertTrue(cfg_a == cfg_b)

    def test_fromfile(self):
        json_path = os.path.join(self.path, 'test_configs.json')
        with open(json_path, 'w') as f:
            f.write(json.dumps(dict(name='a', t=1, _t=2)))
        cfg_a = BaseConfig(json_path)

        cfg_b = BaseConfig(dict(t=1))

        self.assertTrue(cfg_a == cfg_b)


class TestEnv(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(env.getdir(env.paths.test_folder),
                                os.path.splitext(os.path.basename(__file__))[0], cls.__name__)
        if not os.path.exists(cls.path):
            os.makedirs(cls.path)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_getdir(self):
        self.assertTrue(os.path.samefile(env.getdir(env.paths.test_folder), os.path.split(__file__)[0]))

    def test_chdir(self):
        old_cwd = os.getcwd()
        env.chdir(env.paths.test_folder)
        self.assertTrue(os.path.samefile(os.getcwd(), os.path.split(__file__)[0]))
        os.chdir(old_cwd)


class TestRun(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(env.getdir(env.paths.test_folder),
                                os.path.splitext(os.path.basename(__file__))[0], cls.__name__)
        if not os.path.exists(cls.path):
            os.makedirs(cls.path)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_more(self):
        r_a = Run(dict(cuda=True))
        self.assertEqual(r_a.cuda, torch.cuda.is_available())
        r_b = Run(dict(cuda=False))
        self.assertFalse(r_b.cuda)


if __name__ == '__main__':
    unittest.main(verbosity=2)
