import configs


class BaseTest(object):

    def __init__(self, path):
        self.path = path if isinstance(path, list) else [path]

    def run(self):
        for p in self.path:
            if isinstance(p, str) or isinstance(p, dict):
                print("Testing " + str(p) + " ...")
                print(configs.BaseConfig(p))
            else:
                raise TypeError
