---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
  keep-ipynb: true
---

# Isotropic and Anisotropic Kriging {#sec-iso-aniso-kriging}

This chapter illustrates the difference between isotropic and anisotropic Kriging models. The difference is illustrated with the help of the `spotpython` package. Isotropic Kriging models use the same `theta` value for every dimension. Anisotropic Kriging models use different `theta` values for each dimension.

## Example: Isotropic `Spot` Surrogate and the 2-dim Sphere Function {#sec-spot-2d-sphere-iso}


```{python}
import numpy as np
from math import inf
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import Spot
from spotpython.utils.init import fun_control_init, surrogate_control_init
PREFIX="003"
```

### The Objective Function: 2-dim Sphere

The `spotpython` package provides several classes of objective functions. We will use an analytical objective function, i.e., a function that can be described by a (closed) formula:

$$
f(x, y) = x^2 + y^2
$$
The size of the `lower` bound vector determines the problem dimension. Here we will use `np.array([-1, -1])`, i.e., a two-dimensional function.

```{python}
fun = Analytical().fun_sphere
fun_control = fun_control_init(PREFIX=PREFIX,
                               lower = np.array([-1, -1]),
                               upper = np.array([1, 1]))
```

Although the default `spot` surrogate model is an isotropic Kriging model, we will explicitly set the `n_theta` parameter to a value of `1`, so that the same theta value is used for both dimensions.
This is done to illustrate the difference between isotropic and anisotropic Kriging models.

```{python}
surrogate_control=surrogate_control_init(n_theta=1)
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

## Example With Anisotropic Kriging

As described in @sec-spot-2d-sphere-iso, the default parameter setting of `spotpython`'s Kriging surrogate uses the same `theta` value for every dimension. This is referred to as "using an isotropic kernel".  If different `theta` values are used for each dimension, then an anisotropic kernel is used. To enable anisotropic models in `spotpython`, the number of `theta` values should be larger than one. We can use `surrogate_control=surrogate_control_init(n_theta=2)` to enable this behavior (`2` is the problem dimension).

```{python}
surrogate_control = surrogate_control_init(n_theta=2)
spot_2_anisotropic = Spot(fun=fun,
                    fun_control=fun_control,
                    surrogate_control=surrogate_control)
spot_2_anisotropic.run()
```

The search progress of the optimization with the anisotropic model can be visualized:

```{python}
spot_2_anisotropic.plot_progress(log_y=True)
```

```{python}
spot_2_anisotropic.print_results()
```

```{python}
spot_2_anisotropic.surrogate.plot()
```

### Taking a Look at the `theta` Values

#### `theta` Values from the `spot` Model

We can check, whether one or several `theta` values were used. The `theta` values from the surrogate can be printed as follows:

```{python}
spot_2_anisotropic.surrogate.theta
```

* Since the surrogate from the isotropic setting was stored as `spot_2`, we can also take a look at the `theta` value from this model:

```{python}
spot_2.surrogate.theta
```

#### TensorBoard

Now we can start TensorBoard in the background with the following command:



```{raw}
tensorboard --logdir="./runs"
```



We can access the TensorBoard web server with the following URL:



```{raw}
http://localhost:6006/
```



The TensorBoard plot illustrates how `spotpython` can be used as a microscope for the internal mechanisms of the surrogate-based optimization process. Here, one important parameter, the learning rate $\theta$ of the Kriging surrogate is plotted against the number of optimization steps.

![TensorBoard visualization of the spotpython surrogate model.](figures_static/03_tensorboard_03.png){width="100%"}



## Exercises


### 1. The Branin Function `fun_branin`

* Describe the function.
  * The input dimension is `2`. The search range is  $-5 \leq x_1 \leq 10$ and $0 \leq x_2 \leq 15$.
* Compare the results from `spotpython` run a) with isotropic and b) anisotropic surrogate models.
* Modify the termination criterion: instead of the number of evaluations (which is specified via `fun_evals`), the time should be used as the termination criterion. This can be done as follows (`max_time=1` specifies a run time of one minute):

```{python}
#| eval: false
from math import inf
fun_control = fun_control_init(
              fun_evals=inf,
              max_time=1)
```

### 2. The Two-dimensional Sin-Cos Function `fun_sin_cos`

* Describe the function.
  *  The input dimension is `2`. The search range is  $-2\pi \leq x_1 \leq 2\pi$ and $-2\pi \leq x_2 \leq 2\pi$.
* Compare the results from `spotpython` run a) with isotropic and b) anisotropic surrogate models.
* Modify the termination criterion (`max_time` instead of `fun_evals`) as described for `fun_branin`.

### 3. The Two-dimensional Runge Function `fun_runge`

* Describe the function.
  *  The input dimension is `2`. The search range is  $-5 \leq x_1 \leq 5$ and $-5 \leq x_2 \leq 5$.
* Compare the results from `spotpython` run a) with isotropic and b) anisotropic surrogate models.
* Modify the termination criterion (`max_time` instead of `fun_evals`) as described for `fun_branin`.

### 4. The Ten-dimensional Wing-Weight Function `fun_wingwt`

* Describe the function.
  *  The input dimension is `10`. The search ranges are between 0 and 1 (values are mapped internally to their natural bounds).
* Compare the results from `spotpython` run a) with isotropic and b) anisotropic surrogate models.
* Modify the termination criterion (`max_time` instead of `fun_evals`) as described for `fun_branin`.


### 5. The Two-dimensional Rosenbrock Function `fun_rosen` {#sec-09-exercise-rosen}

* Describe the function.
  *  The input dimension is `2`. The search ranges are between -5 and 10.
* Compare the results from `spotpython` run a) with isotropic and b) anisotropic surrogate models.
* Modify the termination criterion (`max_time` instead of `fun_evals`) as described for `fun_branin`.


## Selected Solutions

### Solution to Exercise @sec-09-exercise-rosen: The Two-dimensional Rosenbrock Function `fun_rosen`

#### The Two Dimensional `fun_rosen`: The Isotropic Case

```{python}
import numpy as np
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.init import fun_control_init, surrogate_control_init
from spotpython.spot import Spot
```

The `spotpython` package provides several classes of objective functions.
We will use the `fun_rosen` in the `analytical` class [[SOURCE]](https://github.com/sequential-parameter-optimization/spotpython/blob/main/src/spotpython/fun/objectivefunctions.py).

```{python}
fun_rosen = Analytical().fun_rosen
```


Here we will use problem dimension $k=2$, which can be specified by the `lower` bound arrays.
The size of the `lower` bound array determines the problem dimension.

The prefix is set to `"ROSEN"` to distinguish the results from the one-dimensional case.
Again, TensorBoard can be used to monitor the progress of the optimization.


```{python}
fun_control = fun_control_init(
              PREFIX="ROSEN",
              lower = np.array([-5, -5]),
              upper = np.array([10, 10]),
              show_progress=True)
surrogate_control = surrogate_control_init(n_theta=1)
spot_rosen = Spot(fun=fun_rosen,
                  fun_control=fun_control,
                  surrogate_control=surrogate_control)
spot_rosen.run()
```

::: {.callout-note}
Now we can start TensorBoard in the background with the following command:

```{raw}
tensorboard --logdir="./runs"
```
and can access the TensorBoard web server with the following URL:

```{raw}
http://localhost:6006/
```
::: 

##### Results

```{python}
_ = spot_rosen.print_results()
```

```{python}
spot_rosen.plot_progress()
```

##### A Contour Plot

We can select two dimensions, say $i=0$ and $j=1$, and generate a contour plot as follows.


```{python}
min_z = None
max_z = None
spot_rosen.plot_contour(i=0, j=1, min_z=min_z, max_z=max_z)
```


* The variable importance cannot be calculated, because only one `theta` value was used.


##### TensorBoard

TBD



#### The Two Dimensional `fun_rosen`: The Anisotropic Case


```{python}
import numpy as np
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.init import fun_control_init, surrogate_control_init
from spotpython.spot import Spot
```


The `spotpython` package provides several classes of objective functions.
We will use the `fun_rosen` in the `analytical` class [[SOURCE]](https://github.com/sequential-parameter-optimization/spotpython/blob/main/src/spotpython/fun/objectivefunctions.py).

```{python}
fun_rosen = Analytical().fun_rosen
```


Here we will use problem dimension $k=2$, which can be specified by the `lower` bound arrays.
The size of the `lower` bound array determines the problem dimension. 

We can also add interpreable labels to the dimensions, which will be used in the plots. 
```{python}
fun_control = fun_control_init(
              PREFIX="ROSEN",
              lower = np.array([-5, -5]),
              upper = np.array([10, 10]),
              show_progress=True)
surrogate_control = surrogate_control_init(n_theta=2)
spot_rosen = Spot(fun=fun_rosen,
                  fun_control=fun_control,
                  surrogate_control=surrogate_control)
spot_rosen.run()
```

::: {.callout-note}
Now we can start TensorBoard in the background with the following command:

```{raw}
tensorboard --logdir="./runs"
```
and can access the TensorBoard web server with the following URL:

```{raw}
http://localhost:6006/
```
::: 

##### Results

```{python}
_ = spot_rosen.print_results()
```

```{python}
spot_rosen.plot_progress()
```

##### A Contour Plot

We can select two dimensions, say $i=0$ and $j=1$, and generate a contour plot as follows.


```{python}
min_z = None
max_z = None
spot_rosen.plot_contour(i=0, j=1, min_z=min_z, max_z=max_z)
```


* The variable importance can be calculated as follows:

```{python}
_ = spot_rosen.print_importance()
```


```{python}
spot_rosen.plot_importance()
```

##### TensorBoard

TBD



## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this lecture is available on GitHub in the [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/009_num_spot_anisotropic.ipynb)

:::

