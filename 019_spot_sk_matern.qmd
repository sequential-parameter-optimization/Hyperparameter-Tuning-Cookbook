---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
---

# Benchmarking SPOT Kriging with Matern Kernel on 6D Rosenbrock Function and 10D Michalewicz Function

:::{.callout-note}
These test functions were used during the Dagstuhl Seminar 25451 Bayesian Optimisation (Nov 02 – Nov 07, 2025), see [here](https://www.dagstuhl.de/25451).

:::

## SPOT Kriging in 6 Dimensions:  Rosenbrock Function

This notebook demonstrates how to use the `Spot` class from `spotpython` for Kriging surrogates on the 6-dimensional Rosenbrock function.
We use a maximum of 100 function evaluations.

To visualize the optimization process, you can start tensorboard with:

```bash
tensorboard --logdir="runs/"
```


```{python}
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import Spot
from spotpython.utils.init import fun_control_init, design_control_init, surrogate_control_init
from spotpython.plot.contour import plotModel
```

### Define the 6D Rosenbrock Function

```{python}
dim = 6
lower = np.full(dim, -2)
upper = np.full(dim, 2)
fun = Analytical().fun_rosenbrock
fun_evals = 100
```

### Set up SPOT Controls

```{python}
init_size = dim
seed = 321
max_surrogate_points = fun_evals
max_time = 60
```

### Compile the necessary imports

```{python}
fun_control = fun_control_init(
    lower=lower,
    upper=upper,
    fun_evals=fun_evals,
    seed=seed,
    show_progress=True,
    TENSORBOARD_CLEAN=True,
    tensorboard_log=True,
    max_time=max_time
)
design_control = design_control_init(init_size=init_size)
surrogate_control_exact = surrogate_control_init(max_surrogate_points=max_surrogate_points)
```

### Sklearn Gaussian Process Regressor as Surrogate

```{python}
#| label: kriging-matern-6d-rosen_run
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# Used a Matern kernel instead of the standard spotpython RBF kernel
kernel = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5)
S_GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)

S_rosen = Spot(
    fun=fun,
    fun_control=fun_control,
    design_control=design_control,
    surrogate_control=surrogate_control_exact,
    surrogate=S_GP,
)
S_rosen.run()
```

```{python}
S_rosen.plot_progress(log_y=True, title="Exact sklearn Kriging Progress with y")
```

```{python}
print(f"[6D] Exact Kriging y: min y = {S_rosen.min_y:.4f} at x = {S_rosen.min_X}")
```

Plot of the surrogate model in the first two dimensions:

```{python}
model = S_rosen.surrogate
fig, axes = plotModel(
    model=model,
    lower=lower,
    upper=upper,
    i=0,
    j=1,
    n_grid=100,
    contour_levels=20,
)
```


### Evaluation of 30 repeats with Kriging and Matern kernel

Since 30 repeats were performed and stored in `spot_rosen.json`, we can now evaluate the results:

```{python}
# Load results from spot_rosen.json
with open("spot_rosen.json", "r") as f:
    data = json.load(f)

# Extract all "evaluations" values
evals = [
    iteration["sampled_locations"][0]["evaluations"]
    for iteration in data["search_iterations"]
]

# Compute mean and standard deviation
mean_eval = np.mean(evals)
std_eval = np.std(evals)

print(f"Mean of evaluations: {mean_eval:.6f}")
print(f"Standard deviation of evaluations: {std_eval:.6f}")
```


##  SPOT Kriging in 10 Dimensions:  Exact (Michalewicz Function)


This notebook demonstrates how to use the `Spot` class from `spotpython`  for Kriging surrogates on the 10-dimensional Michalewicz function.
We use a maximum of 300 function evaluations.

### Define the 10D Michalewicz Function

```{python}
dim = 10
lower = np.full(dim, 0)
upper = np.full(dim, np.pi)
fun = Analytical().fun_michalewicz
fun_evals = 300
max_time =  60
```

### Set up SPOT Controls

```{python}
init_size = dim
seed = 321
max_surrogate_points = fun_evals
```

### Compile the necessary imports

```{python}
fun_control = fun_control_init(
    lower=lower,
    upper=upper,
    fun_evals=fun_evals,
    seed=seed,
    show_progress=True,
    TENSORBOARD_CLEAN=True,
    tensorboard_log=True,
    max_time=max_time
)
design_control = design_control_init(init_size=init_size)
surrogate_control_exact = surrogate_control_init(max_surrogate_points=max_surrogate_points)
```

### Sklearn Gaussian Process Regressor as Surrogate

```{python}
#| label: kriging-matern-10d-michalewicz_run
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# Used a Matern kernel instead of the standard spotpython RBF kernel
kernel = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5)
S_GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
S_micha = Spot(
    fun=fun,
    fun_control=fun_control,
    design_control=design_control,
    surrogate_control=surrogate_control_exact,
    surrogate=S_GP,
)
S_micha.run()
```

```{python}
S_micha.plot_progress(log_y=False, title="sklearn Kriging Progress with y")
```

```{python}
print(f"[10D] Kriging y: min y = {S_micha.min_y:.4f} at x = {S_micha.min_X}")
```


Plot of the surrogate model in the first two dimensions:

```{python}
model = S_micha.surrogate
fig, axes = plotModel(
    model=model,
    lower=lower,
    upper=upper,
    i=0,
    j=1,
    n_grid=100,
    contour_levels=20,
)
```


### Evaluation of 30 repeats with Kriging and Matern kernel

Sine 30 repeats were performed and stored in `spot_michalewicz.json`, we can now evaluate the results:

```{python}
# Load results from spot_michalewicz.json
with open("spot_michalewicz.json", "r") as f:
    data = json.load(f)

# Extract all "evaluations" values
evals = [
    iteration["sampled_locations"][0]["evaluations"]
    for iteration in data["search_iterations"]
]

# Compute mean and standard deviation
mean_eval = np.mean(evals)
std_eval = np.std(evals)

print(f"Mean of evaluations: {mean_eval:.6f}")
print(f"Standard deviation of evaluations: {std_eval:.6f}")
```


## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this chapter is available on GitHub in the [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/019_spot_sk_matern.ipynb)

:::


