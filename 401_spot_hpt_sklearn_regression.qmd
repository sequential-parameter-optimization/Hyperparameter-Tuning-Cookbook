---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
---

# HPT: sklearn SVR on Regression Data {#sec-hpt-sklearn-svR}

This chapter is a tutorial for the Hyperparameter Tuning (HPT) of a `sklearn` SVR model on a regression dataset.

## Step 1: Setup {#sec-setup-svr}

Before we consider the detailed experimental setup, we select the parameters that affect run time, initial design size and the device that is used.

::: {.callout-caution}
### Caution: Run time and initial design size should be increased for real experiments

* MAX_TIME is set to one minute for demonstration purposes. For real experiments, this should be increased to at least 1 hour.
* INIT_SIZE is set to 5 for demonstration purposes. For real experiments, this should be increased to at least 10.

:::


```{python}
MAX_TIME = 1
INIT_SIZE = 20
PREFIX = "18"
```

```{python}
#| echo: false
import os
from math import inf
import numpy as np
import warnings
if not os.path.exists('./figures'):
    os.makedirs('./figures')
warnings.filterwarnings("ignore")
```


## Step 2: Initialization of the Empty `fun_control` Dictionary

`spotpython` supports the visualization of the hyperparameter tuning process with TensorBoard. The following example shows how to use TensorBoard with `spotpython`.
The `fun_control` dictionary is the central data structure that is used to control the optimization process. It is initialized as follows:


```{python}
from spotpython.utils.init import fun_control_init
from spotpython.hyperparameters.values import set_control_key_value
from spotpython.utils.eda import print_res_table
fun_control = fun_control_init(
    PREFIX=PREFIX,
    TENSORBOARD_CLEAN=True,
    max_time=MAX_TIME,
    fun_evals=inf,
    tolerance_x = np.sqrt(np.spacing(1)))
```

::: {.callout-tip}
#### Tip: TensorBoard
* Since the `spot_tensorboard_path` argument is not `None`, which is the default, `spotpython` will log the optimization process in the TensorBoard folder.
* The `TENSORBOARD_CLEAN` argument is set to `True` to archive the TensorBoard folder if it already exists. This is useful if you want to start a hyperparameter tuning process from scratch.
If you want to continue a hyperparameter tuning process, set `TENSORBOARD_CLEAN` to `False`. Then the TensorBoard folder will not be archived and the old and new TensorBoard files will shown in the TensorBoard dashboard.
:::


## Step 3: SKlearn Load Data (Classification) {#sec-data-loading-17}

Randomly generate classification data. Here, we use similar data as in [Comparison of kernel ridge regression and SVR](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html#sphx-glr-auto-examples-miscellaneous-plot-kernel-ridge-regression-py).

```{python}
import numpy as np

rng = np.random.RandomState(42)

X = 5 * rng.rand(10, 1)
y = np.sin(1/X).ravel()*np.cos(X).ravel()

# Add noise to targets
y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))

X_plot = np.linspace(0, 5, 100000)[:, None]
```

```{python}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

n_features = 1
target_column = "y"
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
train = pd.DataFrame(np.hstack((X_train, y_train.reshape(-1, 1))))
test = pd.DataFrame(np.hstack((X_test, y_test.reshape(-1, 1))))
train.columns = [f"x{i}" for i in range(1, n_features+1)] + [target_column]
test.columns = [f"x{i}" for i in range(1, n_features+1)] + [target_column]
train.head()
```

```{python}
n_samples = len(train)
# add the dataset to the fun_control
fun_control.update({"data": None, # dataset,
               "train": train,
               "test": test,
               "n_samples": n_samples,
               "target_column": target_column})
```

## Step 4: Specification of the Preprocessing Model {#sec-specification-of-preprocessing-model-17}

Data preprocesssing can be very simple, e.g., you can ignore it. Then you would choose the `prep_model` "None":

```{python}
prep_model = None
fun_control.update({"prep_model": prep_model})
```

A default approach for numerical data is the `StandardScaler` (mean 0, variance 1).  This can be selected as follows:

```{python}
from sklearn.preprocessing import StandardScaler
prep_model = StandardScaler
fun_control.update({"prep_model": prep_model})
```

Even more complicated pre-processing steps are possible, e.g., the follwing pipeline:

```{raw}
categorical_columns = []
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
prep_model = ColumnTransformer(
         transformers=[
             ("categorical", one_hot_encoder, categorical_columns),
         ],
         remainder=StandardScaler,
     )
```

## Step 5: Select Model (`algorithm`) and `core_model_hyper_dict`

The selection of the algorithm (ML model) that should be tuned is done by specifying the its name from the `sklearn` implementation.  For example, the `SVC` support vector machine classifier is selected as follows:

```{python}
from spotpython.hyperparameters.values import add_core_model_to_fun_control
from spotpython.hyperdict.sklearn_hyper_dict import SklearnHyperDict
from sklearn.svm import SVR
add_core_model_to_fun_control(core_model=SVR,
                              fun_control=fun_control,
                              hyper_dict=SklearnHyperDict,
                              filename=None)
```

Now `fun_control` has the information from the JSON file.
The corresponding entries for the `core_model` class are shown below.

```{python}
fun_control['core_model_hyper_dict']
```

:::{.callout-note}
#### `sklearn Model` Selection

The following `sklearn` models are supported by default:

* RidgeCV
* RandomForestClassifier
* SVC
* SVR
* LogisticRegression
* KNeighborsClassifier
* GradientBoostingClassifier
* GradientBoostingRegressor
* ElasticNet

They can be imported as follows:

```{python}
#| eval: false
#| label: 017_import_sklearn_models
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
```

:::


## Step 6: Modify `hyper_dict` Hyperparameters for the Selected Algorithm aka `core_model`

 `spotpython` provides functions for modifying the hyperparameters, their bounds and factors as well as for activating and de-activating hyperparameters without re-compilation of the Python source code. These functions were described in @sec-modifying-hyperparameter-levels.

### Modify hyperparameter of type numeric and integer (boolean)

Numeric and boolean values can be modified using the `modify_hyper_parameter_bounds` method.  

:::{.callout-note}
#### `sklearn Model` Hyperparameters

The hyperparameters of the `sklearn`  `SVC` model are described in the [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

:::


* For example, to change the `tol` hyperparameter of the `SVC` model to the interval [1e-5, 1e-3], the following code can be used:

```{python}
from spotpython.hyperparameters.values import modify_hyper_parameter_bounds
modify_hyper_parameter_bounds(fun_control, "tol", bounds=[1e-5, 1e-3])
modify_hyper_parameter_bounds(fun_control, "epsilon", bounds=[0.1, 1.0])
# modify_hyper_parameter_bounds(fun_control, "degree", bounds=[2, 5])
fun_control["core_model_hyper_dict"]["tol"]
```

### Modify hyperparameter of type factor

Factors can be modified with the `modify_hyper_parameter_levels` function.  For example, to exclude the `sigmoid` kernel from the tuning, the `kernel` hyperparameter of the `SVR` model can be modified as follows:

```{python}
from spotpython.hyperparameters.values import modify_hyper_parameter_levels
# modify_hyper_parameter_levels(fun_control, "kernel", ["poly", "rbf"])
modify_hyper_parameter_levels(fun_control, "kernel", ["rbf"])
fun_control["core_model_hyper_dict"]["kernel"]
```

### Optimizers {#sec-optimizers-17}

Optimizers are described in @sec-optimizer.

## Step 7: Selection of the Objective (Loss) Function

There are two metrics:

1. `metric_river` is used for the river based evaluation via `eval_oml_iter_progressive`.
2. `metric_sklearn` is used for the sklearn based evaluation.

```{python}
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_curve, roc_auc_score, log_loss, mean_squared_error
fun_control.update({
               "metric_sklearn": mean_squared_error,
               "weights": 1.0,
               })
```

:::{.callout-warning}
#### `metric_sklearn`: Minimization and Maximization

* Because the `metric_sklearn` is used for the sklearn based evaluation, it is important to know whether the metric should be minimized or maximized.
* The `weights` parameter is used to indicate whether the metric should be minimized or maximized.
* If `weights` is set to `-1.0`, the metric is maximized.
* If `weights` is set to `1.0`, the metric is minimized, e.g., `weights = 1.0` for `mean_absolute_error`, or `weights = -1.0` for `roc_auc_score`.

:::

### Predict Classes or Class Probabilities

If the key `"predict_proba"` is set to `True`, the class probabilities are predicted. `False` is the default, i.e., the classes are predicted.

```{python}
fun_control.update({
               "predict_proba": False,
               })
```

## Step 8: Calling the SPOT Function


### The Objective Function {#sec-the-objective-function-17}

The objective function is selected next. It implements an interface from `sklearn`'s training, validation, and  testing methods to `spotpython`.

```{python}
from spotpython.fun.hypersklearn import HyperSklearn
fun = HyperSklearn().fun_sklearn
```

The following code snippet shows how to get the default hyperparameters as an array, so that they can be passed to the `Spot` function.

```{python}
from spotpython.hyperparameters.values import get_default_hyperparameters_as_array
X_start = get_default_hyperparameters_as_array(fun_control)
```

### Run the `Spot` Optimizer

The class `Spot` [[SOURCE]](https://github.com/sequential-parameter-optimization/spotpython/blob/main/src/spotpython/spot/spot.py) is the hyperparameter tuning workhorse. It is initialized with the following parameters:

* `fun`: the objective function
* `fun_control`: the dictionary with the control parameters for the objective function
* `design`: the experimental design
* `design_control`: the dictionary with the control parameters for the experimental design
* `surrogate`: the surrogate model
* `surrogate_control`: the dictionary with the control parameters for the surrogate model
* `optimizer`: the optimizer
* `optimizer_control`: the dictionary with the control parameters for the optimizer

:::{.callout-note}
#### Note: Total run time
 The total run time may exceed the specified `max_time`, because the initial design (here: `init_size` = INIT_SIZE as specified above) is always evaluated, even if this takes longer than `max_time`.
:::

```{python}
from spotpython.utils.init import design_control_init, surrogate_control_init
design_control = design_control_init()
set_control_key_value(control_dict=design_control,
                        key="init_size",
                        value=INIT_SIZE,
                        replace=True)

surrogate_control = surrogate_control_init(method="regression",
                                           n_theta=2)
from spotpython.spot import Spot
spot_tuner = Spot(fun=fun,
                   fun_control=fun_control,
                   design_control=design_control,
                   surrogate_control=surrogate_control)
spot_tuner.run(X_start=X_start)
```

### TensorBoard {#sec-tensorboard-17}

Now we can start TensorBoard in the background with the following command, where `./runs` is the default directory for the TensorBoard log files:

```{raw}
tensorboard --logdir="./runs"
```


```{python}
from spotpython.utils.init import get_tensorboard_path
get_tensorboard_path(fun_control)
```

After the hyperparameter tuning run is finished, the progress of the hyperparameter tuning can be visualized. The black points represent the performace values (score or metric) of  hyperparameter configurations from the initial design, whereas the red points represents the  hyperparameter configurations found by the surrogate model based optimization.

```{python}
spot_tuner.plot_progress(log_y=True)
```

Results can also be printed in tabular form.

```{python}
print_res_table(spot_tuner)
```

A histogram can be used to visualize the most important hyperparameters.

```{python}
spot_tuner.plot_importance(threshold=0.0025)
```

## Get Default Hyperparameters

The default hyperparameters, which will be used for a comparion with the tuned hyperparameters, can be obtained with the following commands:

```{python}
from spotpython.hyperparameters.values import get_one_core_model_from_X
from spotpython.hyperparameters.values import get_default_hyperparameters_as_array
X_start = get_default_hyperparameters_as_array(fun_control)
model_default = get_one_core_model_from_X(X_start, fun_control, default=True)
model_default
```

## Get SPOT Results

In a similar way, we can obtain the hyperparameters found by `spotpython`.

```{python}
from spotpython.hyperparameters.values import get_one_core_model_from_X
X_tuned = spot_tuner.to_all_dim(spot_tuner.min_X.reshape(1,-1))
model_spot = get_one_core_model_from_X(X_tuned, fun_control)
```

### Plot: Compare Predictions

```{python}
model_default.fit(X_train, y_train)
y_default = model_default.predict(X_plot)
```

```{python}
model_spot.fit(X_train, y_train)
y_spot = model_spot.predict(X_plot)
```

```{python}
import matplotlib.pyplot as plt
plt.scatter(X[:100], y[:100], c="orange", label="data", zorder=1, edgecolors=(0, 0, 0))
plt.plot(
    X_plot,
    y_default,
    c="red",
    label="Default SVR")

plt.plot(
    X_plot, y_spot, c="blue", label="SPOT SVR")

plt.xlabel("data")
plt.ylabel("target")
plt.title("SVR")
_ = plt.legend()
```

### Detailed Hyperparameter Plots

```{python}
spot_tuner.plot_important_hyperparameter_contour(filename=None)
```

### Parallel Coordinates Plot

```{python}
spot_tuner.parallel_plot()
```

