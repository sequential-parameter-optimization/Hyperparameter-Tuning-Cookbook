---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
---


## Kriging with Varying Correlation-p {#sec-num-spot-correlation-p}


This chapter illustrates the difference between Kriging models with varying p. The difference is illustrated with the help of the `spotpython` package. 

## Example: `Spot` Surrogate and the 2-dim Sphere Function


```{python}
import numpy as np
from math import inf
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import Spot
from spotpython.utils.init import fun_control_init, surrogate_control_init
PREFIX="015"
```

### The Objective Function: 2-dim Sphere

* The `spotpython` package provides several classes of objective functions.
* We will use an analytical objective function, i.e., a function that can be described by a (closed) formula:
   $$f(x, y) = x^2 + y^2$$
* The size of the `lower` bound vector determines the problem dimension.
* Here we will use `np.array([-1, -1])`, i.e., a two-dim function.

```{python}
fun = Analytical().fun_sphere
fun_control = fun_control_init(PREFIX=PREFIX,
                               lower = np.array([-1, -1]),
                               upper = np.array([1, 1]))
```

* Although the default `spot` surrogate model is an isotropic Kriging model, we will explicitly set the `theta` parameter to a value of `1` for both dimensions. This is done to illustrate the difference between isotropic and anisotropic Kriging models.

```{python}
surrogate_control=surrogate_control_init(n_p=1,
                                         p_val=2.0,)
```

```{python}
spot_2 = Spot(fun=fun,
                   fun_control=fun_control,
                   surrogate_control=surrogate_control)

spot_2.run()
```

### Results

```{python}
spot_2.print_results()
```

```{python}
spot_2.plot_progress(log_y=True)
```

```{python}
spot_2.surrogate.plot()
```

## Example With Modified p

* We can use set `p_val` to a value other than `2` to obtain a different Kriging model.

```{python}
surrogate_control = surrogate_control_init(n_p=1,
                                           p_val=1.0)
spot_2_p1= Spot(fun=fun,
                    fun_control=fun_control,
                    surrogate_control=surrogate_control)
spot_2_p1.run()
```

* The search progress of the optimization with the anisotropic model can be visualized:

```{python}
spot_2_p1.plot_progress(log_y=True)
```

```{python}
spot_2_p1.print_results()
```

```{python}
spot_2_p1.surrogate.plot()
```

### Taking a Look at the `p_val` Values

#### `p_val` Values from the `spot` Model

* We can check, which `p_val` values the `spot` model has used:
* The `p_val` values from the surrogate can be printed as follows:

```{python}
spot_2_p1.surrogate.p_val
```

* Since the surrogate from the isotropic setting was stored as `spot_2`, we can also take a look at the `theta` value from this model:

```{python}
spot_2.surrogate.p_val
```

## Optimization of the `p_val` Values

```{python}
surrogate_control = surrogate_control_init(n_p=1,
                                           optim_p=True)
spot_2_pm= Spot(fun=fun,
                    fun_control=fun_control,
                    surrogate_control=surrogate_control)
spot_2_pm.run()
```

```{python}
spot_2_pm.plot_progress(log_y=True)
```

```{python}
spot_2_pm.print_results()
```

```{python}
spot_2_pm.surrogate.plot()
```

```{python}
spot_2_pm.surrogate.p_val
```

## Optimization of Multiple `p_val` Values

```{python}
surrogate_control = surrogate_control_init(n_p=2,
                                           optim_p=True)
spot_2_pmo= Spot(fun=fun,
                    fun_control=fun_control,
                    surrogate_control=surrogate_control)
spot_2_pmo.run()
```

```{python}
spot_2_pmo.plot_progress(log_y=True)
```

```{python}
spot_2_pmo.print_results()
```

```{python}
spot_2_pmo.surrogate.plot()
```

```{python}
spot_2_pmo.surrogate.p_val
```



## Exercises


###  `fun_branin`

* Describe the function.
  * The input dimension is `2`. The search range is  $-5 \leq x_1 \leq 10$ and $0 \leq x_2 \leq 15$.
* Compare the results from `spotpython` runs with different options for `p_val`.
* Modify the termination criterion: instead of the number of evaluations (which is specified via `fun_evals`), the time should be used as the termination criterion. This can be done as follows (`max_time=1` specifies a run time of one minute):

```{python}
fun_evals=inf,
max_time=1,
```

### `fun_sin_cos`

* Describe the function.
  *  The input dimension is `2`. The search range is  $-2\pi \leq x_1 \leq 2\pi$ and $-2\pi \leq x_2 \leq 2\pi$.
* Compare the results from `spotpython` run a) with isotropic and b) anisotropic surrogate models.
* Modify the termination criterion (`max_time` instead of `fun_evals`) as described for `fun_branin`.

###  `fun_runge`

* Describe the function.
  *  The input dimension is `2`. The search range is  $-5 \leq x_1 \leq 5$ and $-5 \leq x_2 \leq 5$.
* Compare the results from `spotpython` runs with different options for `p_val`.
* Modify the termination criterion (`max_time` instead of `fun_evals`) as described for `fun_branin`.

###  `fun_wingwt`

* Describe the function.
  *  The input dimension is `10`. The search ranges are between 0 and 1 (values are mapped internally to their natural bounds).
* Compare the results from `spotpython` runs with different options for `p_val`.
* Modify the termination criterion (`max_time` instead of `fun_evals`) as described for `fun_branin`.




## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this lecture is available on GitHub in the [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/015_num_spot_correlation_p.ipynb)

:::

