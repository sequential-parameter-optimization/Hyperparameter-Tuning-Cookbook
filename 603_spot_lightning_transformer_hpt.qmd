---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
jupyter: python3
---

# Hyperparameter Tuning of a Transformer Network with PyTorch Lightning {#sec-hyperparameter-tuning-with-pytorch-lightning-603}

## Basic Setup {#sec-basic-setup-603}

This section provides an overview of the hyperparameter tuning process using `spotpython` and `PyTorch` Lightning. It uses the `Diabetes` data set (see @sec-a-05-diabetes-data-set) for a regression task. 

```{python}
#| echo: false
#| label: 603_imports
import numpy as np
import os
from math import inf
import numpy as np
import warnings
if not os.path.exists('./figures'):
    os.makedirs('./figures')
warnings.filterwarnings("ignore")
```

In this section, we will show how `spotpython` can be integrated into the `PyTorch` Lightning
training workflow for a regression task.
It demonstrates how easy it is to use `spotpython` to tune hyperparameters for a `PyTorch` Lightning model.

After importing the necessary libraries, the `fun_control` dictionary is set up via the `fun_control_init` function.
The `fun_control` dictionary contains

* `PREFIX`: a unique identifier for the experiment
* `fun_evals`: the number of function evaluations
* `max_time`: the maximum run time in minutes
* `data_set`: the data set. Here we use the `Diabetes` data set that is provided by `spotpython`.
* `core_model_name`: the class name of the neural network model. This neural network model is provided by `spotpython`.
* `hyperdict`: the hyperparameter dictionary. This dictionary is used to define the hyperparameters of the neural network model. It is also provided by `spotpython`.
* `_L_in`: the number of input features. Since the `Diabetes` data set has 10 features, `_L_in` is set to 10.
* `_L_out`: the number of output features. Since we want to predict a single value, `_L_out` is set to 1.

The method `set_hyperparameter` allows the user to modify default hyperparameter settings. Here we set the `initialization` method to `["Default"]`. No other initializations are used in this experiment.
The `HyperLight` class is used to define the objective function `fun`. It connects the `PyTorch` and the `spotpython` methods and is provided by `spotpython`.
Finally, a `Spot` object is created.

Note, the `divergence_threshold` is set to 25,000, which is based on some pre-experiments with the `Diabetes` data set.

```{python}
#| label: 603_setup
from spotpython.data.diabetes import Diabetes
from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.fun.hyperlight import HyperLight
from spotpython.utils.init import (fun_control_init, surrogate_control_init, design_control_init)
from spotpython.utils.eda import print_exp_table
from spotpython.hyperparameters.values import set_hyperparameter
from spotpython.spot import Spot
from spotpython.utils.file import get_experiment_filename
from spotpython.utils.scaler import TorchStandardScaler

fun_control = fun_control_init(
    PREFIX="603",
    TENSORBOARD_CLEAN=True,
    tensorboard_log=True,
    fun_evals=inf,
    max_time=1,
    data_set = Diabetes(),
    scaler=TorchStandardScaler(),
    core_model_name="light.regression.NNTransformerRegressor",
    hyperdict=LightHyperDict,
    divergence_threshold=25_000,
    _L_in=10,
    _L_out=1)

set_hyperparameter(fun_control, "optimizer", [
                "Adadelta",
                "Adagrad",
                "Adam",
                "AdamW",
                "Adamax",
            ])
set_hyperparameter(fun_control, "epochs", [5, 7])
set_hyperparameter(fun_control, "nhead", [1, 2])
set_hyperparameter(fun_control, "dim_feedforward_mult", [1, 1])

design_control = design_control_init(init_size=5)
surrogate_control = surrogate_control_init(
    method="regression",
    min_Lambda=1e-3,
    max_Lambda=10,
)

fun = HyperLight().fun

spot_tuner = Spot(fun=fun,fun_control=fun_control, design_control=design_control, surrogate_control=surrogate_control)
```

We can take a look at the design table to see the initial design.
```{python}
#| label: 603_design_table
print_exp_table(fun_control)
```


Calling the method `run()` starts the hyperparameter tuning process on the local machine.

```{python}
#| label: 603_run
res = spot_tuner.run()
```

Note that we have enabled Tensorboard-Logging, so we can visualize the results with Tensorboard. Execute the
following command in the terminal to start Tensorboard.

```{python}
#| label: 603_tensorboard
#| eval: false
tensorboard --logdir="runs/"
```


## Looking at the Results

### Tuning Progress

After the hyperparameter tuning run is finished, the progress of the hyperparameter tuning can be visualized with `spotpython`'s method `plot_progress`. The black points represent the performace values (score or metric) of  hyperparameter configurations from the initial design, whereas the red points represents the  hyperparameter configurations found by the surrogate model based optimization.

```{python}
#| label: 603_plot_progress
spot_tuner.plot_progress(log_y=True, filename=None)
```

### Tuned Hyperparameters and Their Importance

Results can be printed in tabular form.

```{python}
#| label: 603_gen_design_table_results
from spotpython.utils.eda import print_res_table
print_res_table(spot_tuner)
```

## Hyperparameter Considerations

1. `d_model` (or `d_embedding`):

   - This is the dimension of the embedding space or the number of expected features in the input.
   - All input features are projected into this dimensional space before entering the transformer encoder.
   - This dimension must be divisible by `nhead` since each head in the multi-head attention mechanism will process a subset of `d_model/nhead` features.

2. `nhead`:

   - This is the number of attention heads in the multi-head attention mechanism.
   - It allows the transformer to jointly attend to information from different representation subspaces.
   - It's important that `d_model % nhead == 0` to ensure the dimensions are evenly split among the heads.

3. `num_encoder_layers`:

   - This specifies the number of transformer encoder layers stacked together.
   - Each layer contains a multi-head attention mechanism followed by position-wise feedforward layers.

4. `dim_feedforward`:

   - This is the dimension of the feedforward network model within the transformer encoder layer.
   - Typically, this dimension is larger than `d_model` (e.g., 2048 for a Transformer model with `d_model=512`).

### Important: Constraints and Interconnections:

- `d_model` and `nhead`:
  - As mentioned, `d_model` must be divisible by `nhead`. This is critical because each attention head operates simultaneously on a part of the embedding, so `d_model/nhead` should be an integer.

- `num_encoder_layers` and `dim_feedforward`**: 
  - These parameters are more flexible and can be chosen independently of `d_model` and `nhead`.
  - However, the choice of `dim_feedforward` does influence the computational cost and model capacity, as larger dimensions allow learning more complex representations. 

- One hyperparameter does not strictly need to be a multiple of others except for ensuring `d_model % nhead == 0`.

### Practical Considerations:

1. Setting `d_model`:

   - Common choices for `d_model` are powers of 2 (e.g., 256, 512, 1024).
   - Ensure that it matches the size of the input data after the linear projection layer.

2. Setting `nhead`:

   - Typically, values are 1, 2, 4, 8, etc., depending on the `d_model` value.
   - Each head works on a subset of features, so `d_model / nhead` should be large enough to be meaningful.

3. Setting `num_encoder_layers`:

   - Practical values range from 1 to 12 or more depending on the depth desired.
   - Deeper models can capture more complex patterns but are also more computationally intensive.

4. Setting `dim_feedforward`:

   - Often set to a multiple of `d_model`, such as 2048 when `d_model` is 512.
   - Ensures sufficient capacity in the intermediate layers for complex feature transformations.


::: {.callout-note}
### Note: `d_model` Calculation 

Since `d_model % nhead == 0` is a critical constraint to ensure that the multi-head attention mechanism can operate effectively, `spotpython` computes the value of `d_model` based on the `nhead` value provided by the user. This ensures that the hyperparameter configuration is valid. So, the final value of `d_model` is a multiple of `nhead`.
`spotpython` uses the hyperparameter `d_model_mult` to determine the multiple of `nhead` to use for `d_model`, i.e., `d_model = nhead * d_model_mult`.
:::

::: {.callout-note}
### Note: `dim_feedforward` Calculation

Since this dimension is typically larger than `d_model` (e.g., 2048 for a Transformer model with `d_model=512`),
`spotpython` uses the hyperparameter `dim_feedforward_mult` to determine the multiple of `d_model` to use for `dim_feedforward`, i.e., `dim_feedforward = d_model * dim_feedforward_mult`.

::: 

## Summary

This section presented an introduction to the basic setup of hyperparameter tuning of a transformer with `spotpython` and `PyTorch` Lightning.