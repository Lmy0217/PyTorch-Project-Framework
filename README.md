# PyTorch-Project-Framework
[![Travis](https://img.shields.io/travis/Lmy0217/PyTorch-Project-Framework.svg?branch=master&label=Travis+CI)](https://www.travis-ci.org/Lmy0217/PyTorch-Project-Framework) [![CircleCI](https://img.shields.io/circleci/project/github/Lmy0217/PyTorch-Project-Framework.svg?branch=master&label=CircleCI)](https://circleci.com/gh/Lmy0217/PyTorch-Project-Framework) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Lmy0217/PyTorch-Project-Framework/pulls)

A high cohesion, low coupling, and plug-and-play project framework for PyTorch.


## Folder Structure
```
  ├── configs
  |    ├── BaseConfig.py  - the loader of all configuration file
  |    ├── BaseTest.py  - the test class of all configuration file
  |    ├── Env.py  - the loader of environmental configuration file
  |    └── Run  - the loader of hyperparameter configuration file
  |
  ├── datasets
  |    ├── BaseDataset.py  - the abstract class of all dataset
  |    ├── BaseTest.py  - the test class of all dataset
  |    └── ...  - any dataset of your project
  |
  ├── models
  |    ├── BaseModel.py  - the abstract class of all model
  |    ├── BaseTest.py  - the test class of all model
  |    └── ...  - any model of your project
  |
  ├── res
  |    ├── env  - the folder contains any json file of environmental configuration
  |    ├── datasets  - the folder contains any json file of dataset configuration
  |    ├── models  - the folder contains any json file of model configuration
  |    └── run  - the folder contains any json file of hyperparameter configuration
  |
  ├── utils
  |    └── logger.py  - the logger class
  |
  ├── main.py  - the main class of framework
  |
  └── test.py  - the global test class
```


## Main Components

### Datasets
- Base dataset
Base dataset is an abstract class that must be Inherited by any dataset you create, the idea behind this is that there's much shared stuff between all datasets. The base dataset mainly contains:
  - `more`  - add / update unique configuration to dataset
  - `_load`  - load dataset
  - `_recover`  - split single data
  - `split`  - create trainset and testset


- Your dataset
Here's where you implement your dataset. So you should:
  - Create your dataset class and inherit the `BaseDataset` class
  - Override `_load` method
  - Override other methods if your need special implementation
  - Add your dataset name to `datasets/__init__.py`
  - Create json file of your dataset's configuration in `res/datasets/`

### Models
- Base model
Base model is an abstract class that must be Inherited by any model you create, the idea behind this is that there's much shared stuff between all models. The base model mainly contains:
  - `check_cfg`  - filter data set
  - `train`  - train step
  - `test`  - test step
  - `load`  - load previously trained model
  - `save`  - save model


- Your model
Here's where you implement your model. So you should:
  - Create your model class and inherit the `BaseModel` class
  - Override `train` / `test` method
  - Override other methods if your need special implementation
  - Add your model name to `models/__init__.py`
  - Create json file of your model's configuration in `res/models/`


## How to Use
Here's how to use this framework, you should do the following:

- Dataset
	- In `datasets` folder create a class that inherit the "BaseDataset" class
		```python
	    # YourDataset.py
		class YourDataset(datasets.BaseDataset):
		    def __init__(self, cfg, **kwargs):
		    super(YourDataset, self).__init__(cfg, **kwargs)
		
		```
	- Override `_load` method to load dataset
		```python
	    # In YourDataset class
		def _load(self):
	        """
	        Here load your dataset
            The parameters in `cfg` are load from json file of your dataset's configuration
	        For example:
	        - Create 4 random images of size (depth, height, width) as source data 
	        - Create 4 random labels as target data
	        Return data dictionary and the amount of data
	        """

	        data_count = 4
	        source = np.random.rand(data_count, self.cfg.depth, self.cfg.height, self.cfg.width)
	        target = np.random.randint(0, self.cfg.label_count, (data_count, 1))

	        return {'source': source, 'target': target}, data_count
		
		```
	- Add your dataset name to `datasets/__init__.py`
	    ```python
	    from .YourDataset import YourDataset
	    ```
	- Create json file of your dataset's configuration in `res/datasets/`
	    ```json
	    {
	        "name": "YourDataset", // same with your dataset class name
	        // All dataset parameter your need where create `YourDataset` class
	        // For example, the size of images and K-fold cross-validation
	        "source": {
	            depth: 3,
	            height: 128,
	            width: 128
	        },
            "cross_folds": 2
	    }
        ```

- Model
	- In `models` folder create a class that inherit the "BaseModel" class
		```python
	    # YourModel.py
		class YourModel(models.BaseModels):
            def __init__(self, cfg, data_cfg, run, **kwargs):
            super(YourModel, self).__init__(cfg, data_cfg, run, **kwargs)

            # The parameters in `cfg` are load from json file of your model's configuration
            # The parameters in `data_cfg` are load from json file of dataset's configuration
            # The parameters in `run` are load from json file of hyperparameter configuration

            # Create model, optimizer, criterion, and etc.
            # For example:
            # - model: Linear
            # - criterion: L1 loss
            # - optimizer: Adam
            self.model = nn.Linear(self.cfg.input_dims, self.cfg.output_dims).to(self.device)
            self.criterion = nn.L1Loss.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.run.lr, betas=(self.run.b1, self.run.b2))
        ```
    - Override two methods `train` and `test` to write the logic of the training and testing process
        ```python
        # In YourDataset class
        def train(self, batch_idx, sample_dict):
            """
            batch_idx: the index of batch
            sample_dict: the dictionary of train data

            Implement the logic of training process
            For example:
                source -> [model] -> predict -> [criterion] (+target) -> loss
            Return loss dictionary
            """
            source = sample_dict['source'].to(self.device)
            target = sample_dict['target'].to(self.device)

            self.model.train()
            self.optimizer.zero_grad()
            predict = self.model(source)
            loss = self.criterion(predict, target)
            loss.backward()
            self.optimizer.step()

            # Others you need to calculate

            return {'loss': loss}

        def test(self, batch_idx, sample_dict):
            """
            batch_idx: the index of batch
            sample_dict: the dictionary of test data

            Implement the logic of testing process
            For example:
                source -> [model] -> predict
            Return dictionary of data which you want saved
            """
            source = sample_dict['source'].to(self.device)
            target = sample_dict['target'].to(self.device)

            self.model.eval()
            predict = self.model(source)

            # Others you need to calculate

            return {'target': target, 'predict': predict}
        ```
	- Add your model name to `models/__init__.py`
	    ```python
	    from .YourModel import YourModel
	    ```
	- Create json file of your model's configuration in `res/models/`
	    ```json
	    {
	        "name": "YourModel", // same with your model class name
	        // All model parameter your need where create `YourModel` class
	        // For example, the dimensions of input and output
	        "input_dims": 256,
            "output_dims": 1
	    }
        ```

- Hyperparameter
    - Create json file of your hyperparameter's configuration in `res/run/`
	    ```json
	    {
	        "name": "hp1",
            // Basic hyperparameter
            "batch_size": 32,
            "epochs": 200,
            "save_step": 10,
	        // Hyperparameters your need where create optimizer in `YourModel` class or others
	        // For example, learning rate
	        "lr": 2e-4
	    }
        ```

- Run `main.py` to start training or testing
    - Training with configuration files `res/datasets/yourdataset.json`, `res/models/yourmodel.json`, and `res/run/yourhp.json`
	    ```bash
	    python3 -m main --dataset_config_path "res/datasets/yourdataset.json" --model_config_path "res/models/yourmodel.json" --run_config_path "res/run/hp.json"
	    ```
    Every `save_step` epoch trained model and data which want to saved will be saved in the folder `save/[yourmodel]-[yourhp]-[yourdataset]-[index of cross-validation]`.

    - If you want to testing epoch 10
	    ```bash
	    python3 -m main --dataset_config_path "res/datasets/yourdataset.json" --model_config_path "res/models/yourmodel.json" --run_config_path "res/run/hp.json" --test_epoch 10
	    ```

## Contributing
Any kind of enhancement or contribution is welcomed.


## License
The code is licensed with the [MIT](LICENSE) license.