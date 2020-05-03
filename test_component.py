import os
import models
import configs
import datasets


class Test(object):

    def __init__(self):
        self.ci = configs.env.ci.run
        configs.env.ci.run = True

    def __del__(self):
        configs.env.ci.run = self.ci

    def _path(self, path):
        full_path = configs.env.getdir(path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        configs.env.chdir(path)

    def run(self):
        self.test_models()
        self.test_configs()
        self.test_datasets()

    def test_models(self):
        self._path('models')
        for model in models.BaseModel.__subclasses__():
            models.BaseTest(model).run()

    def test_configs(self):
        self._path('res')
        files = [os.path.join(os.getcwd(), file) for file in os.listdir(os.getcwd())]
        index = 0
        while index < len(files):
            if os.path.isdir(files[index]):
                files.extend([os.path.join(files[index], f) for f in os.listdir(files[index])])
                files.remove(files[index])
            else:
                index += 1
        for file in files:
            configs.BaseTest(file).run()

    def test_datasets(self):
        self._path('datasets')
        for dataset in datasets.BaseDataset.__subclasses__():
            datasets.BaseTest(dataset).run()


if __name__ == "__main__":
    Test().run()
