import json


class BaseConfig(object):

    def __init__(self, cfg, **kwargs):
        self._space = 0
        self._load(cfg) if isinstance(cfg, dict) else self._fromfile(cfg)

    def _values(self, value: str):
        return not (value.startswith('_') or value == 'name')

    def dict(self):
        value_dict = dict(vars(self))
        for _value in list(value_dict.keys()):
            if not self._values(_value):
                value_dict.pop(_value, None)
        return value_dict

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            flag = True
            for name, value in vars(self).items():
                flag &= (value == getattr(other, name))
            return flag
        else:
            return False

    def __repr__(self):
        s, sp = '', ' ' * self._space
        if self._space == 0:
            s += self.__class__.__name__
            if hasattr(self, 'name'):
                s += ' (' + str(self.name) + ')'
            s += ': '
        s += '{\n'
        v = list(vars(self).items())
        v.sort()
        for name, value in v:
            if not self._values(name):
                continue
            s += sp + '  ' + name + ': '
            if value.__class__.__base__ == BaseConfig or value.__class__ == BaseConfig:
                value._space = self._space + 2
            if isinstance(value, str):
                s += "'"
            s += str(value)
            if isinstance(value, str):
                s += "'"
            s += '\n'
        s += sp + '}'
        if self._space == 0:
            s += '\n'
        return s

    def _load(self, config_dict):
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                setattr(self, key, BaseConfig(value) if isinstance(value, dict) else value)

    def _fromfile(self, path):
        with open(path, 'r') as f:
            self._load(json.load(f))
        setattr(self, '_path', path)
