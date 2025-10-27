---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
jupyter: python3
---

# Hyperparameter Tuning with PyTorch Lightning and User Models {#sec-light-user-model-601}

```{python}
#| echo: false
#| label: 601_user_model_first_imports
import numpy as np
import os
from math import inf
import numpy as np
import warnings
if not os.path.exists('./figures'):
    os.makedirs('./figures')
warnings.filterwarnings("ignore")
```

In this section, we will show how a user defined model can be used for the `PyTorch` Lightning hyperparameter tuning workflow with `spotpython`.

## Using a User Specified Model

As templates, we provide the following three files that allow the user to specify a model in the `/userModel` directory:

* `my_regressor.py`, see @sec-my-regressor
* `my_hyperdict.json`, see @sec-my-hyper-dict-json
* `my_hyperdict.py`, see @sec-my-hyper-dict.

The `my_regressor.py` file contains the model class, which is a subclass of `nn.Module`.
The `my_hyperdict.json` file contains the hyperparameter settings as a dictionary, which are loaded via the `my_hyperdict.py` file.

Note, that we have to add the path to the `userModel` directory to the `sys.path` list as shown below.
The `divergence_threshold` is set to 3,000, which is based on some pre-experiments with the user data set.

```{python}
#| label: 601_user_model_imports_sys
import sys
sys.path.insert(0, './userModel')
import my_regressor
import my_hyper_dict
from spotpython.hyperparameters.values import add_core_model_to_fun_control

from spotpython.data.diabetes import Diabetes
from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.fun.hyperlight import HyperLight
from spotpython.utils.init import (fun_control_init, design_control_init)
from spotpython.utils.eda import print_res_table
from spotpython.hyperparameters.values import set_hyperparameter
from spotpython.spot import Spot

fun_control = fun_control_init(
    PREFIX="601-user-model",
    fun_evals=inf,
    max_time=1,
    data_set = Diabetes(),
    divergence_threshold=3_000,
    _L_in=10,
    _L_out=1)

add_core_model_to_fun_control(fun_control=fun_control,
                              core_model=my_regressor.MyRegressor,
                              hyper_dict=my_hyper_dict.MyHyperDict)

design_control = design_control_init(init_size=7)

fun = HyperLight().fun

spot_tuner = Spot(fun=fun,fun_control=fun_control, design_control=design_control)
```


```{python}
#| label: 601_user_model_run
res = spot_tuner.run()
print_res_table(spot_tuner)
spot_tuner.plot_important_hyperparameter_contour(max_imp=3)
```

## Details

### Model Setup

By using `core_model_name = "my_regressor.MyRegressor"`, the user specified model class `MyRegressor` [[SOURCE]](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/userModel/my_regressor.py) is selected.
For this given `core_model_name`, the local hyper_dict is loaded using the `my_hyper_dict.py` file as shown below.

### The `my_hyper_dict.py` File {#sec-my-hyper-dict}

The `my_hyper_dict.py` file  must be placed in the `/userModel` directory.  It provides a convenience function to load the hyperparameters from user specified the `my_hyper_dict.json` file, see @sec-my-hyper-dict.
The user does not need to modify this file, if the JSON file is stored as `my_hyper_dict.json`. 
Alternative filenames can be specified via the `filename` argument (which is default set to `"my_hyper_dict.json"`).

### The `my_hyper_dict.json` File {#sec-my-hyper-dict-json}

The `my_hyper_dict.json` file contains the hyperparameter settings as a dictionary, which are loaded via the `my_hyper_dict.py` file.
The example below shows the content of the `my_hyper_dict.json` file.
```json
{
    "MyRegressor": {
        "l1": {
            "type": "int",
            "default": 3,
            "transform": "transform_power_2_int",
            "lower": 3,
            "upper": 8
        },
        "epochs": {
            "type": "int",
            "default": 4,
            "transform": "transform_power_2_int",
            "lower": 4,
            "upper": 9
        },
        "batch_size": {
            "type": "int",
            "default": 4,
            "transform": "transform_power_2_int",
            "lower": 1,
            "upper": 4
        },
        "act_fn": {
            "levels": [
                "Sigmoid",
                "Tanh",
                "ReLU",
                "LeakyReLU",
                "ELU",
                "Swish"
            ],
            "type": "factor",
            "default": "ReLU",
            "transform": "None",
            "class_name": "spotpython.torch.activation",
            "core_model_parameter_type": "instance()",
            "lower": 0,
            "upper": 5
        },
        "optimizer": {
            "levels": [
                "Adadelta",
                "Adagrad",
                "Adam",
                "AdamW",
                "SparseAdam",
                "Adamax",
                "ASGD",
                "NAdam",
                "RAdam",
                "RMSprop",
                "Rprop",
                "SGD"
            ],
            "type": "factor",
            "default": "SGD",
            "transform": "None",
            "class_name": "torch.optim",
            "core_model_parameter_type": "str",
            "lower": 0,
            "upper": 11
        },
        "dropout_prob": {
            "type": "float",
            "default": 0.01,
            "transform": "None",
            "lower": 0.0,
            "upper": 0.25
        },
        "lr_mult": {
            "type": "float",
            "default": 1.0,
            "transform": "None",
            "lower": 0.1,
            "upper": 10.0
        },
        "patience": {
            "type": "int",
            "default": 2,
            "transform": "transform_power_2_int",
            "lower": 2,
            "upper": 6
        },
        "initialization": {
            "levels": [
                "Default",
                "Kaiming",
                "Xavier"
            ],
            "type": "factor",
            "default": "Default",
            "transform": "None",
            "core_model_parameter_type": "str",
            "lower": 0,
            "upper": 2
        }
    }
}
```

### The `my_regressor.py` File {#sec-my-regressor}

The `my_regressor.py` file contains [[SOURCE]](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/userModel/my_regressor.py) the model class, which is a subclass of `nn.Module`. It must implement the following methods:

* `__init__(self, **kwargs)`: The constructor of the model class. The hyperparameters are passed as keyword arguments.
* `forward(self, x: torch.Tensor) -> torch.Tensor`: The forward pass of the model. The input `x` is passed through the model and the output is returned.
* `training_step(self, batch, batch_idx) -> torch.Tensor`: The training step of the model. It takes a batch of data and the batch index as input and returns the loss.
* `validation_step(self, batch, batch_idx) -> torch.Tensor`: The validation step of the model. It takes a batch of data and the batch index as input and returns the loss.
* `test_step(self, batch, batch_idx) -> torch.Tensor`: The test step of the model. It takes a batch of data and the batch index as input and returns the loss.
* `predict(self, x: torch.Tensor) -> torch.Tensor`: The prediction method of the model. It takes an input `x` and returns the prediction.
* `configure_optimizers(self) -> torch.optim.Optimizer`: The method to configure the optimizer of the model. It returns the optimizer.

The file `my_regressor.py` must be placed in the `/userModel` directory. The user can modify the model class to implement a custom model architecture.

We will take a closer look at the methods defined in the `my_regressor.py` file in the next subsections.

#### The `__init__` Method

`__init__()` initializes the `MyRegressor` object. It takes the following arguments:

* `l1` (int): The number of neurons in the first hidden layer.
* `epochs` (int): The number of epochs to train the model for.
* `batch_size` (int): The batch size to use during training.
* `initialization` (str): The initialization method to use for the weights.
* `act_fn` (nn.Module): The activation function to use in the hidden layers.
* `optimizer` (str): The optimizer to use during training.
* `dropout_prob` (float): The probability of dropping out a neuron during training.
* `lr_mult` (float): The learning rate multiplier for the optimizer.
* `patience` (int): The number of epochs to wait before early stopping.
* `_L_in` (int): The number of input features. Not a hyperparameter, but needed to create the network.
* `_L_out` (int): The number of output classes. Not a hyperparameter, but needed to create the network.
* `_torchmetric` (str): The metric to use for the loss function. If `None`, then "mean_squared_error" is used.

It is implemented as follows:

```{python}
#| label: 601_user_model_init
#| eval: false
class MyRegressor(L.LightningModule):
        def __init__(
        self,
        l1: int,
        epochs: int,
        batch_size: int,
        initialization: str,
        act_fn: nn.Module,
        optimizer: str,
        dropout_prob: float,
        lr_mult: float,
        patience: int,
        _L_in: int,
        _L_out: int,
        _torchmetric: str,
        *args,
        **kwargs,
    ):
        super().__init__()
        self._L_in = _L_in
        self._L_out = _L_out
        if _torchmetric is None:
            _torchmetric = "mean_squared_error"
        self._torchmetric = _torchmetric
        self.metric = getattr(torchmetrics.functional.regression, _torchmetric)
        # _L_in and _L_out are not hyperparameters, but are needed to create the network
        # _torchmetric is not a hyperparameter, but is needed to calculate the loss
        self.save_hyperparameters(ignore=["_L_in", "_L_out", "_torchmetric"])
        # set dummy input array for Tensorboard Graphs
        # set log_graph=True in Trainer to see the graph (in traintest.py)
        self.example_input_array = torch.zeros((batch_size, self._L_in))
        if self.hparams.l1 < 4:
            raise ValueError("l1 must be at least 4")
        hidden_sizes = [l1 * 2, l1, ceil(l1/2)]
        # Create the network based on the specified hidden sizes
        layers = []
        layer_sizes = [self._L_in] + hidden_sizes
        layer_size_last = layer_sizes[0]
        for layer_size in layer_sizes[1:]:
            layers += [
                nn.Linear(layer_size_last, layer_size),
                self.hparams.act_fn,
                nn.Dropout(self.hparams.dropout_prob),
            ]
            layer_size_last = layer_size
        layers += [nn.Linear(layer_sizes[-1], self._L_out)]
        # nn.Sequential summarizes a list of modules into a single module,
        # applying them in sequence
        self.layers = nn.Sequential(*layers)
```

#### The `hidden_sizes`

`__init__()` uses the `hidden_sizes` list to define the sizes of the hidden layers in the network. The hidden sizes are calculated based on the `l1` hyperparameter. The hidden sizes can be computed in different ways, depending on the problem and the desired network architecture. We recommend using a separate function to calculate the hidden sizes based on the hyperparameters.

#### The `forward` Method

The `forward()` method defines the forward pass of the model. It takes an input tensor `x` and passes it through the network layers to produce an output tensor. It is implemented as follows:

```{python}
#| label: 601_user_model_forward
#| eval: false
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers(x)
```

#### The `_calculate_loss` Method

The `_calculate_loss()` method calculates the loss based on the predicted output and the target values. It uses the specified metric to calculate the loss. 
It takes the following arguments:

* `batch (tuple)`: A tuple containing a batch of input data and labels.

It is implemented as follows:

```{python}
#| label: 601_user_model_calculate_loss
#| eval: false
def _calculate_loss(self, batch):
    x, y = batch
    y = y.view(len(y), 1)
    y_hat = self(x)
    loss = self.metric(y_hat, y)
    return loss
``` 

#### The `training_step` Method

The `training_step()` method defines the training step of the model. It takes a batch of data and returns the loss. It is implemented as follows:

```{python}
#| label: 601_user_model_training_step
#| eval: false
def training_step(self, batch: tuple) -> torch.Tensor:
    val_loss = self._calculate_loss(batch)
    return val_loss
```

#### The `validation_step` Method

The `validation_step()` method defines the validation step of the model. It takes a batch of data and returns the loss. It is implemented as follows:

```{python}
#| label: 601_user_model_validation_step
#| eval: false
def validation_step(self, batch: tuple) -> torch.Tensor:
    val_loss = self._calculate_loss(batch)
    return val_loss
```

#### The `test_step` Method

The `test_step()` method defines the test step of the model. It takes a batch of data and returns the loss. It is implemented as follows:

```{python}
#| label: 601_user_model_test_step
#| eval: false
def test_step(self, batch: tuple) -> torch.Tensor:
    val_loss = self._calculate_loss(batch)
    return val_loss
```

#### The `predict` Method

The `predict()` method defines the prediction method of the model. It takes an input tensor `x` and returns 
a tuple with the input tensor `x`, the target tensor `y`, and the predicted tensor `y_hat`.

 It is implemented as follows:

```{python}
#| label: 601_user_model_predict
#| eval: false
def predict(self, x: torch.Tensor) -> torch.Tensor:
    x, y = batch
    yhat = self(x)
    y = y.view(len(y), 1)
    yhat = yhat.view(len(yhat), 1)
    return (x, y, yhat)
```

#### The `configure_optimizers` Method

The `configure_optimizers()` method defines the optimizer to use during training.
It uses the `optimizer_handler` from `spotpython.hyperparameter.optimizer` to create the optimizer based on the specified optimizer name, parameters, and learning rate multiplier.
It is implemented as follows:

```{python}
#| label: 601_user_model_configure_optimizers
#| eval: false
def configure_optimizers(self) -> torch.optim.Optimizer:
    optimizer = optimizer_handler(
        optimizer_name=self.hparams.optimizer, params=self.parameters(), lr_mult=self.hparams.lr_mult
    )
    return optimizer
```

Note, the default Lightning way is to define an optimizer as
`optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)`.
`spotpython` uses an optimizer handler to create the optimizer, which adapts the learning rate according to the `lr_mult` hyperparameter as
well as other hyperparameters. See `spotpython.hyperparameters.optimizer.py` [[SOURCE]](https://github.com/sequential-parameter-optimization/spotPython/blob/main/src/spotpython/hyperparameters/optimizer.py) for details.

## Connection with the LightDataModule

The steps described in @sec-my-regressor are connected to the `LightDataModule` class [[DOC]](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/data/lightdatamodule/).
This class is used to create the data loaders for the training, validation, and test sets.
The `LightDataModule` class is part of the `spotpython` package and class provides the following methods [[SOURCE]](https://github.com/sequential-parameter-optimization/spotPython/blob/main/src/spotpython/data/lightdatamodule.py):

* `prepare_data()`: This method is used to prepare the data set.
* `setup()`: This method is used to create the data loaders for the training, validation, and test sets.
* `train_dataloader()`: This method is used to return the data loader for the training set.
* `val_dataloader()`: This method is used to return the data loader for the validation set.
* `test_dataloader()`: This method is used to return the data loader for the test set.
* `predict_dataloader()`: This method is used to return the data loader for the prediction set.

### The `prepare_data()` Method

The `prepare_data()` method is used to prepare the data set.
This method is called only once and on a single process.
It can be used to download the data set. In our case, the data set is already available, so this method uses a simple `pass` statement.

### The `setup()` Method

The `stage` is used to define the data set to be returned. It 
can be `None`, `fit`, `test`, or `predict`.
If `stage` is `None`, the method returns the training (`fit`),
testing (`test`), and prediction (`predict`) data sets.

The `setup` methods splits the data based on the `stage` setting for use in training, validation, and testing.
It uses `torch.utils.data.random_split()` to split the data.

Splitting is based on the `test_size` and `test_seed`. 
The `test_size` can be a float or an int.

First, the data set sizes are determined as described in @sec-determine-sizes-601.
Then, the data sets are split based on the `stage` setting.
`spotpython`'s `LightDataModule` class uses the following sizes:

* `full_train_size`: The size of the full training data set. This data set is splitted into the final training data set and a validation data set.
* `val_size`: The size of the validation data set. The validation data set is used to validate the model during training.
* `train_size`: The size of the training data set. The training data set is used to train the model.
* `test_size`: The size of the test data set. The test data set is used to evaluate the model after training. It is not used during training ("hyperparameter tuning"). Only after everything is finished, the model is evaluated on the test data set.

#### Determine the Sizes of the Data Sets {#sec-determine-sizes-601}

```{python}
#| label: 601_user_data_module_setup
import torch
from torch.utils.data import random_split
data_full = Diabetes()
test_size = fun_control["test_size"]
test_seed=fun_control["test_seed"]
# if test_size is float, then train_size is 1 - test_size
if isinstance(test_size, float):
    full_train_size = round(1.0 - test_size, 2)
    val_size = round(full_train_size * test_size, 2)
    train_size = round(full_train_size - val_size, 2)
else:
    # if test_size is int, then train_size is len(data_full) - test_size
    full_train_size = len(data_full) - test_size
    val_size = int(full_train_size * test_size / len(data_full))
    train_size = full_train_size - val_size

print(f"LightDataModule setup(): full_train_size: {full_train_size}")
print(f"LightDataModule setup(): val_size: {val_size}")
print(f"LightDataModule setup(): train_size: {train_size}")
print(f"LightDataModule setup(): test_size: {test_size}")
```


#### The "setup" Method: Stage "fit" {#sec-stage-fit-601}

Here, `train_size` and `val_size` are used to split the data into training and validation sets.

```{python}
#| label: 601_user_data_module_setup_fit
stage = "fit"
scaler = None
# Assign train/val datasets for use in dataloaders
if stage == "fit" or stage is None:
    print(f"train_size: {train_size}, val_size: {val_size} used for train & val data.")
    generator_fit = torch.Generator().manual_seed(test_seed)
    data_train, data_val, _ = random_split(
        data_full, [train_size, val_size, test_size], generator=generator_fit
    )
    if scaler is not None:
        # Fit the scaler on training data and transform both train and val data
        scaler_train_data = torch.stack([data_train[i][0] for i in range(len(data_train))]).squeeze(1)
        # train_val_data = data_train[:,0]
        print(scaler_train_data.shape)
        scaler.fit(scaler_train_data)
        data_train = [(scaler.transform(data), target) for data, target in data_train]
        data_tensors_train = [data.clone().detach() for data, target in data_train]
        target_tensors_train = [target.clone().detach() for data, target in data_train]
        data_train = TensorDataset(
            torch.stack(data_tensors_train).squeeze(1), torch.stack(target_tensors_train)
        )
        # print(data_train)
        data_val = [(scaler.transform(data), target) for data, target in data_val]
        data_tensors_val = [data.clone().detach() for data, target in data_val]
        target_tensors_val = [target.clone().detach() for data, target in data_val]
        data_val = TensorDataset(torch.stack(data_tensors_val).squeeze(1), torch.stack(target_tensors_val))
```

The `data_train` and `data_val` data sets are further used to create the training and validation data loaders as 
described in @sec-train-dataloader-601 and @sec-val-dataloader-601, respectively.

#### The "setup" Method: Stage "test" {#sec-stage-test-601}

Here, the test data set, which is based on the `test_size`, is created.

```{python}
#| label: 601_user_data_module_setup_test
stage = "test"
# Assign test dataset for use in dataloader(s)
if stage == "test" or stage is None:
    print(f"test_size: {test_size} used for test dataset.")
    # get test data set as test_abs percent of the full dataset
    generator_test = torch.Generator().manual_seed(test_seed)
    data_test, _ = random_split(data_full, [test_size, full_train_size], generator=generator_test)
    if scaler is not None:
        data_test = [(scaler.transform(data), target) for data, target in data_test]
        data_tensors_test = [data.clone().detach() for data, target in data_test]
        target_tensors_test = [target.clone().detach() for data, target in data_test]
        data_test = TensorDataset(
            torch.stack(data_tensors_test).squeeze(1), torch.stack(target_tensors_test)
        )
print(f"LightDataModule setup(): Test set size: {len(data_test)}")
# Set batch size for DataLoader
batch_size = 5
# Create DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
# Iterate over the data in the DataLoader
for batch in dataloader:
    inputs, targets = batch
    print(f"Batch Size: {inputs.size(0)}")
    print(f"Inputs Shape: {inputs.shape}")
    print(f"Targets Shape: {targets.shape}")
    print("---------------")
    print(f"Inputs: {inputs}")
    print(f"Targets: {targets}")
    break
```

#### The "setup" Method: Stage "predict" {#sec-stage-predict-601}

Prediction and testing use the same data set.
The prediction data set is created based on the `test_size`.

```{python}
#| label: 601_user_data_module_setup_predict
stage = "predict"
if stage == "predict" or stage is None:
    print(f"test_size: {test_size} used for predict dataset.")
    # get test data set as test_abs percent of the full dataset
    generator_predict = torch.Generator().manual_seed(test_seed)
    data_predict, _ = random_split(
        data_full, [test_size, full_train_size], generator=generator_predict
    )
    if scaler is not None:
        data_predict = [(scaler.transform(data), target) for data, target in data_predict]
        data_tensors_predict = [data.clone().detach() for data, target in data_predict]
        target_tensors_predict = [target.clone().detach() for data, target in data_predict]
        data_predict = TensorDataset(
            torch.stack(data_tensors_predict).squeeze(1), torch.stack(target_tensors_predict)
        )
print(f"LightDataModule setup(): Predict set size: {len(data_predict)}")
# Set batch size for DataLoader
batch_size = 5
# Create DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(data_predict, batch_size=batch_size, shuffle=False)
# Iterate over the data in the DataLoader
for batch in dataloader:
    inputs, targets = batch
    print(f"Batch Size: {inputs.size(0)}")
    print(f"Inputs Shape: {inputs.shape}")
    print(f"Targets Shape: {targets.shape}")
    print("---------------")
    print(f"Inputs: {inputs}")
    print(f"Targets: {targets}")
    break
```

### The `train_dataloader()` Method {#sec-train-dataloader-601}

The method ``train_dataloader` returns the training dataloader, i.e., a Pytorch DataLoader instance using the training dataset.
It simply returns a DataLoader with the `data_train` set that was created in the `setup()` method as described in @sec-stage-fit-601.

```{python}
#| eval: false
def train_dataloader(self) -> DataLoader:
    return DataLoader(data_train, batch_size=batch_size, num_workers=num_workers)
```

::: {.callout-note}
#### Using the `train_dataloader()` Method

The `train_dataloader()` method can be used as follows:

```{python}
from spotpython.data.lightdatamodule import LightDataModule
from spotpython.data.diabetes import Diabetes
dataset = Diabetes(target_type=torch.float)
data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.4)
data_module.setup()
print(f"Training set size: {len(data_module.data_train)}")
dl = data_module.train_dataloader()
# Iterate over the data in the DataLoader
for batch in dl:
    inputs, targets = batch
    print(f"Batch Size: {inputs.size(0)}")
    print(f"Inputs Shape: {inputs.shape}")
    print(f"Targets Shape: {targets.shape}")
    print("---------------")
    print(f"Inputs: {inputs}")
    print(f"Targets: {targets}")
    break
```
:::

### The `val_dataloader()` Method {#sec-val-dataloader-601}

Returns the validation dataloader, i.e., a Pytorch DataLoader instance using the validation dataset.
It simply returns a DataLoader with the `data_val` set that was created in the `setup()` method as desccribed in @sec-stage-fit-601.

```{python}
#| eval: false
def val_dataloader(self) -> DataLoader:
    return DataLoader(data_val, batch_size=batch_size, num_workers=num_workers)
```

::: {.callout-note}
#### Using the `val_dataloader()` Method

The `val_dataloader()` method can be used as follows:

```{python}
from spotpython.data.lightdatamodule import LightDataModule
from spotpython.data.diabetes import Diabetes
dataset = Diabetes(target_type=torch.float)
data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.4)
data_module.setup()
print(f"Validation set size: {len(data_module.data_val)}")
dl = data_module.val_dataloader()
# Iterate over the data in the DataLoader
for batch in dl:
    inputs, targets = batch
    print(f"Batch Size: {inputs.size(0)}")
    print(f"Inputs Shape: {inputs.shape}")
    print(f"Targets Shape: {targets.shape}")
    print("---------------")
    print(f"Inputs: {inputs}")
    print(f"Targets: {targets}")
    break
```

::: 


### The `test_dataloader()` Method

Returns the test dataloader, i.e., a Pytorch DataLoader instance using the test dataset.
It simply returns a DataLoader with the `data_test` set that was created in the `setup()` method as described in @sec-stage-test-30.

```{python}
#| eval: false
def test_dataloader(self) -> DataLoader:
    return DataLoader(data_test, batch_size=batch_size, num_workers=num_workers)
```

::: {.callout-note}
#### Using the `test_dataloader()` Method

The `test_dataloader()` method can be used as follows:

```{python}
from spotpython.data.lightdatamodule import LightDataModule
from spotpython.data.diabetes import Diabetes
dataset = Diabetes(target_type=torch.float)
data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.4)
data_module.setup()
print(f"Test set size: {len(data_module.data_test)}")
dl = data_module.test_dataloader()
# Iterate over the data in the DataLoader
for batch in dl:
    inputs, targets = batch
    print(f"Batch Size: {inputs.size(0)}")
    print(f"Inputs Shape: {inputs.shape}")
    print(f"Targets Shape: {targets.shape}")
    print("---------------")
    print(f"Inputs: {inputs}")
    print(f"Targets: {targets}")
    break
```

::: 

### The `predict_dataloader()` Method

Returns the prediction dataloader, i.e., a Pytorch DataLoader instance using the prediction dataset.
It simply returns a DataLoader with the `data_predict` set that was created in the `setup()` method as described in @sec-stage-predict-30.

::: {.callout-warning}
The `batch_size` is set to the length of the `data_predict` set.
:::
```{python}
#| eval: false
def predict_dataloader(self) -> DataLoader:
    return DataLoader(data_predict, batch_size=len(data_predict), num_workers=num_workers)
``` 


::: {.callout-note}
#### Using the `predict_dataloader()` Method

The `predict_dataloader()` method can be used as follows:

```{python}
from spotpython.data.lightdatamodule import LightDataModule
from spotpython.data.diabetes import Diabetes
dataset = Diabetes(target_type=torch.float)
data_module = LightDataModule(dataset=dataset, batch_size=5, test_size=0.4)
data_module.setup()
print(f"Test set size: {len(data_module.data_predict)}")
dl = data_module.predict_dataloader()
# Iterate over the data in the DataLoader
for batch in dl:
    inputs, targets = batch
    print(f"Batch Size: {inputs.size(0)}")
    print(f"Inputs Shape: {inputs.shape}")
    print(f"Targets Shape: {targets.shape}")
    print("---------------")
    print(f"Inputs: {inputs}")
    print(f"Targets: {targets}")
    break
```
:::

## Using the `LightDataModule` in the `train_model()` Method

The methods discussed so far are used in `spotpython`'s  `train_model()` method [[DOC]](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/light/trainmodel/) to train the model.
It is implemented as follows [[SOURCE]](https://github.com/sequential-parameter-optimization/spotPython/blob/main/src/spotpython/light/trainmodel.py).

First, a `LightDataModule` object is created and the `setup()` method is called.
```{python}
#| eval: false
dm = LightDataModule(
    dataset=fun_control["data_set"],
    batch_size=config["batch_size"],
    num_workers=fun_control["num_workers"],
    test_size=fun_control["test_size"],
    test_seed=fun_control["test_seed"],
)
dm.setup()
```

Then, the `Trainer` is initialized.
```{python}
#| eval: false
# Init trainer
trainer = L.Trainer(
    default_root_dir=os.path.join(fun_control["CHECKPOINT_PATH"], config_id),
    max_epochs=model.hparams.epochs,
    accelerator=fun_control["accelerator"],
    devices=fun_control["devices"],
    logger=TensorBoardLogger(
        save_dir=fun_control["TENSORBOARD_PATH"],
        version=config_id,
        default_hp_metric=True,
        log_graph=fun_control["log_graph"],
    ),
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=config["patience"], mode="min", strict=False, verbose=False)
    ],
    enable_progress_bar=enable_progress_bar,
)
```

Next, the `fit()` method is called to train the model.

```{python}
#| eval: false
# Pass the datamodule as arg to trainer.fit to override model hooks :)
trainer.fit(model=model, datamodule=dm)
```

Finally, the `validate()` method is called to validate the model.
The `validate()` method returns the validation loss.

```{python}
#| eval: false
# Test best model on validation and test set
result = trainer.validate(model=model, datamodule=dm)
# unlist the result (from a list of one dict)
result = result[0]
return result["val_loss"]
```


## The Last Connection: The `HyperLight` Class

The method `train_model()` is part of the `HyperLight` class [[DOC]](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/light/trainmodel/). It is called from `spotpython` as an objective function to train the model and return the validation loss.

The `HyperLight` class is implemented as follows [[SOURCE]](https://github.com/sequential-parameter-optimization/spotPython/blob/main/src/spotpython/fun/hyperlight.py).

```{python}
#| eval: false
#| label: 601_user_hyperlight

class HyperLight:
    def fun(self, X: np.ndarray, fun_control: dict = None) -> np.ndarray:
        z_res = np.array([], dtype=float)
        self.check_X_shape(X=X, fun_control=fun_control)
        var_dict = assign_values(X, get_var_name(fun_control))
        for config in generate_one_config_from_var_dict(var_dict, fun_control):
            df_eval = train_model(config, fun_control)
            z_val = fun_control["weights"] * df_eval
            z_res = np.append(z_res, z_val)
        return z_res
```




## Further Information 

### Preprocessing {#sec-preprocessing-601}

Preprocessing is handled by `Lightning` and `PyTorch`. It is described in the [LIGHTNINGDATAMODULE](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) documentation. Here you can find information about the `transforms` methods.