import datasets


class BaseTest(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def run(self):
        for dataset_cfg in datasets.allcfgs():
            if hasattr(dataset_cfg, 'name') and dataset_cfg.name == self.dataset.__name__:
                print('Testing dataset: ' + self.dataset.__name__ + ' ...')
                dataset = self.dataset(dataset_cfg)
                trainset, testset = dataset.split(
                    index_cross=min(dataset.cfg.cross_folder, 1) if hasattr(dataset.cfg, 'cross_folder') else None)

                for splitset, set_name in zip([trainset, testset], ['Trainset', 'Testset']):
                    print("-- " + set_name + " size: " + str(len(splitset)))
                    for i in range(len(splitset)):
                        print("  -- The " + str(i + 1) + "-th sample:", end="")
                        sample_dict, index = splitset[i]
                        for name, value in sample_dict.items():
                            if (hasattr(value, 'ndim') and value.ndim > 1) or value.shape[0] > 1:
                                print(" " + name + " size: ", end="")
                                print(value.shape, end="")
                            else:
                                print(" " + name + " : ", end="")
                                print(value, end="")
                        print('')
                print('')
