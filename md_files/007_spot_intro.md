---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
jupyter: python3
---

# Introduction to Sequential Parameter Optimization {#sec-spot-intro}

The following libraries are used in this document:


```{python}
import numpy as np
from math import inf
from scipy.optimize import shgo
from scipy.optimize import direct
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import Spot
from spotpython.utils.init import (fun_control_init, design_control_init, surrogate_control_init, optimizer_control_init)
from spotpython.data.diabetes import Diabetes
from spotpython.hyperdict.light_hyper_dict import LightHyperDict
from spotpython.fun.hyperlight import HyperLight
from spotpython.utils.eda import print_exp_table, print_res_table
```

This presents an introduction to `spotpythons`'s `Spot` class. The official `spotpython` documentation can be found here: [https://sequential-parameter-optimization.github.io/spotpython/](https://sequential-parameter-optimization.github.io/spotpython/).


## Surrogate Model Based Optimization {#sec-spot}

Surrogate model based optimization methods are common approaches in simulation and optimization. The sequential parameter optimization toolbox (SPOT) was developed because there is a great need for sound statistical analysis of simulation and optimization algorithms [@BLP05 ]. SPOT includes methods for tuning based on classical regression and analysis of variance techniques.
It presents tree-based models such as classification and regression trees and random forests as well as Bayesian optimization (Gaussian process models, also known as Kriging). Combinations of different meta-modeling approaches are possible. SPOT comes with a sophisticated surrogate model based optimization method, that can handle discrete and continuous inputs. Furthermore, any model implemented in `scikit-learn` can be used out-of-the-box as a surrogate in `spotpython`.

SPOT implements key techniques such as exploratory fitness landscape analysis and sensitivity analysis. It can be used to understand the performance of various algorithms, while simultaneously giving insights into their algorithmic behavior.

The `spot` loop consists of the following steps:

1. Init: Build initial design $X$  using the `design` method, e.g., Latin Hypercube Sampling (LHS).
2. Evaluate initial design on real objective $f$: $y = f(X)$ using the `fun` method, e.g., `fun_sphere`.
3. Build surrogate: $S = S(X,y)$, using the `fit` method of the surrogate model, e.g., `Kriging`.
4. Optimize on surrogate: $X_0 =  \text{optimize}(S)$, using the `optimizer` method, e.g., `differential_evolution`.
5. Evaluate on real objective: $y_0 = f(X_0)$, using the `fun` method from above.
6. Impute (Infill) new points: $X = X \cup X_0$, $y = y \cup y_0$, using  `Spot`'s `infill` method.
7. Goto 3.


<!-- \begin{algorithm}
    \caption{Surrogate-Based Optimization}
    \label{alg:spot}
    \begin{algorithmic}[1]
        \State \textbf{Init:} Build initial design $X$
        \State \textbf{Evaluate} initial design on real objective $f$: $y = f(X)$
        \While{stopping criterion not met}
            \State \textbf{Build surrogate:} $S = S(X,y)$
            \State \textbf{Optimize on surrogate:} $X_0 =  \text{optimize}(S)$
            \State \textbf{Evaluate on real objective:} $y_0 = f(X_0)$
            \State \textbf{Impute (Infill) new points:} $X = X \cup X_0$, $y = y \cup y_0$
        \EndWhile
    \end{algorithmic} 
\end{algorithm} -->


## Advantages of the `spotpython` Approach

* Neural networks and many ML algorithms are non-deterministic, so results are noisy (i.e., depend on the the initialization of the weights). Enhanced noise handling strategies, OCBA (description from HPT-book).

* Optimal Computational Budget Allocation (OCBA) is a very efficient solution to solve the "general ranking and selection problem" if the objective function is noisy. It allocates function evaluations in an uneven manner to identify the best solutions and to reduce the total optimization costs. [Chen10a, Bart11b]
Given a total number of optimization samples $N$ to be allocated to 
$k$ competing solutions whose performance is depicted by random variables with means
$\bar{y}_i$ ($i=1, 2, \ldots, k$), and finite variances $\sigma_i^2$, respectively, as 
$N \to \infty$, the \gls{APCS} can be asymptotically maximized when
\begin{align}
\frac{N_i}{N_j} & = \left( \frac{ \sigma_i / \delta_{b,i}}{\sigma_j/ \delta_{b,j}} \right)^2, i,j \in \{ 1, 2, \ldots, k\}, \text{ and }
i \neq j \neq b,\\
N_b &= \sigma_b \sqrt{ 
\sum_{i=1, i\neq b}^k \frac{N_i^2}{\sigma_i^2}
},
\end{align}
where $N_i$ is the number of replications allocated to solution $i$, $\delta_{b,i} = \bar{y}_b - \bar{y}_i$,
and $\bar{y}_b \leq \min_{i\neq b} \bar{y}_i$ [@Chen10a, @Bart11a].


* Surrogate-based optimization: Better than grid search and random search (Reference to HPT-book)
* Visualization
* Importance based on the Kriging model
* Sensitivity analysis.  Exploratory fitness landscape analysis. Provides XAI methods (feature importance, integrated gradients, etc.)
* Uncertainty quantification
* Flexible, modular meta-modeling handling. `spotpython` comes with a Kriging model, which can be replaced by any model implemented in `scikit-learn`.
* Enhanced metric handling, especially for categorical hyperparameters (any `sklearn` metric can be used). Default is..
* Integration with `TensorBoard`: Visualization of the hyperparameter tuning process, of the training steps, the model graph. Parallel coordinates plot, scatter plot matrix, and more.
* Reproducibility. Results are stored as pickle files. The results can be loaded and visualized at any time and be transferred between different machines and operating systems.
* Handles `scikit-learn` models and `PyTorch` models out-of-the-box. The user has to add a simple wrapper for passing the hyperparameters to use a `PyTorch` model in `spotpython`.
* Compatible with `Lightning`.
* User can add own models as plain python code.
* User can add own data sets in various formats.
* Flexible data handling and data preprocessing.
* Many examples online (hyperparameter-tuning-cookbook).
* `spotpython` uses a robust optimizer that can even deal with hyperparameter-settings that cause crashes of the algorithms to be tuned.
* even if the optimum is not found, HPT with `spotpython` prevents the user from choosing bad hyperparameters in a systematic way (design of experiments).


## Disadvantages of the spotpython Approach

* Time consuming
* Surrogate can be misguiding


## An Initial Example: Optimization of the One-dimensional Sphere Function

### The Objective Function: Sphere

The `spotpython` package provides several classes of objective functions. We will use the `sphere`-function, which is an analytical objective function. Analytical functions can be described by a (closed) formula:
$$
f(x) = x^2.
$$

```{python}
#| label: fig-sphere-fun
#| fig-cap: "The sphere function $f(x) = x^2$ in the range [-1, 1]."
fun = Analytical().fun_sphere
x = np.linspace(-1, 1, 100).reshape(-1,1)
y = fun(x)
plt.figure()
plt.plot(x,y, "k")
plt.grid()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Sphere Function")
plt.show()
```

The optimal solution of the sphere function is $x^* = 0$ with $f(x^*) = 0$.

### The `Spot` Method as an Optimization Algorithm Using a Surrogate Model


`pyspot` implements the `Spot` method.
The `Spot` method uses a surrogate model to approximate the objective function and to find the optimal solution by performing an optimization on the surrogate model. In this example, `Spot` uses ten initial design points, which are sampled from a Latin Hypercube Sampling (LHS) design. The `Spot` method will then build a surrogate model based on these initial design points and the corresponding objective function values. The surrogate model is then used to find the optimal solution.
As a default, `Spot` uses a Kriging surrogate model, which is a Gaussian process model. As a default, ten initial design points are sampled from a Latin Hypercube Sampling (LHS) design. Also as a default, 15 function evaluations are performed, i.e., the final surrogate model is built based on 15 points.

The specification of the `lower` and `upper` bounds of the input space is mandatory via the `fun_control` dictionary, is passed to the `Spot` method.
For convenience, `spotpython` provides an initialization method `fun_control_init()` to create the `fun_control` dictionary.
After the `Spot` method is initialized, the `run()` method is called to start the optimization process. The `run()` method will perform the optimization on the surrogate model and return the optimal solution. Finally, the `print_results()` method is called to print the results of the optimization process, i.e., the best objective function value ("min y") and the corresponding input value ("x0").
```{python}
fun_control=fun_control_init(lower = np.array([-1]),
                             upper = np.array([1]))
S = Spot(fun=fun, fun_control=fun_control)
S.run()
S.print_results()
```

`spotpython` provides a method `plot_progress()` to visualize the search progress. The parameter `log_y` is used to plot the objective function values on a logarithmic scale. The black points represent the ten initial design points, whereas the remainin g five red points represent the points found by the surrogate model based optimization.

```{python}
S.plot_progress(log_y=True)
```

```{python}
S.surrogate_control
```


## A Second Example: Optimization of a User-Specified Multi-dimensional Function {#sec-user-specified-fun-branin}

Users can easily specify their own objective functions. The following example shows how to use the `Spot` method to optimize a user-specified two-dimensional function.
Here we will use the 2-dimensional Branin function, which is a well-known test function in optimization. The Branin function is defined as follows:
$$
f(x) = a (x2 - bx1^2 + cx1 - r)^2 + s(1-t) \cos(x1) + s
$$
where:

* $a = 1$,
* $b = 5.1/(4\pi^2)$,
* $c = 5/\pi$,
* $r = 6$,
* $s = 10$, and
* $t = 1/(8\pi)$.

The input space is defined as $x_1 \in [-5, 10]$ and $x_2 \in [0, 15]$.
The user specified Branin function can be implemented as follows (note, that the function is vectorized, i.e., it can handle multiple input points at once and that the `**kwargs` argument is mandatory for compatibility with `spotpython` to pass additional parameters to the function, if needed):

```{python}
def user_fun(X, **kwargs):
    x1 = X[:, 0]
    x2 = X[:, 1]
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    y = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
    return y
```

The Branin function has its global minimum at three points:
$f(x^*) = 0.397887$, at $x^* = (-\pi, 12.275)$, $(\pi, 2.275)$ and $(9.42478, 2.475)$.
It can be visualized as shown in @fig-branin-fun.
```{python}
#| label: fig-branin-fun
#| fig-cap: "The Branin function $f(x)$ in the range $x1 \\in [-5, 10]$ and $x2 \\in [0, 15]$."
x1 = np.linspace(-5, 10, 100)
x2 = np.linspace(0, 15, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = user_fun(np.array([X1.ravel(), X2.ravel()]).T)
Z = Z.reshape(X1.shape)
plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, Z, levels=30, cmap='jet')
plt.colorbar(label='f(x)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Branin Function')
plt.scatter([-np.pi, np.pi, 9.42478], [12.275, 2.275, 2.475], color='white', label='Global Minima')
plt.legend()
plt.show()
```

The default `Spot` class assumes a noisy objective function, because many real-world applications have noisy objective functions. Here, we will set the `noise` parameter to `False` in the `fun_control` dictionary, because the Branin function is a deterministic function. Accordingly, the `interpolation` surrogate method is used, which is suitable for deterministic functions. Therefore, we need to specify the `interpolation` method in the `surrogate_control` dictionary. Furthermore, since the Branin is harder to optimize than the sphere function, we will increase the number of function evaluations to 25 and the number of initial design points to 15.

```{python}
fun_control = fun_control_init(
              lower = np.array( [-5, 0]),
              upper = np.array([10, 15]),
              fun_evals = 25,
              noise = False,
              seed=123
)

design_control=design_control_init(init_size=15)
surrogate_control=surrogate_control_init(method="interpolation")

S_2 = Spot(fun=user_fun,
                 fun_control=fun_control,
                 design_control=design_control,
                 surrogate_control=surrogate_control)
S_2.run()
S_2.print_results()
S_2.plot_progress(log_y=True)
```

The resulting surrogate used by `Spot` can be visualized as shown in @fig-spot-branin-surrogate.

```{python}
#| label: fig-spot-branin-surrogate
#| fig-cap: "Visualization of the surrogate model of the Branin function."
S_2.plot_contour()
```



```{python}
S_2.surrogate_control
```


## A Third Example: Hyperparameter Tuning of a Neural Network

`spotpython` provides several `PyTorch` neural networks, which can be used out-of-the-box for hyperparameter tuning. We will use the `Diabetes` dataset, which is a regression dataset with 10 input features and one output feature.

Similar to the steps from above, we define the `fun_control` dictionary. The hyperparameter tuning of neural networks requires a few additional parameters in the `fun_control` dictionary, e.g., for selecting the neural network model, the hyperparameter dictionary, and the number of input and output features of the neural network. The `fun_control` dictionary is initialized using the `fun_control_init()` method. The following parameters are used:

* The `core_model_name` parameter specifies the neural network model to be used. In this case, we will use the `NNLinearRegressor`, which is a simple linear regression model implemented in `spotpython`.
* The `hyperdict` parameter specifies the hyperparameter dictionary to be used. In this case, we will use the `LightHyperDict`, which is a hyperparameter dictionary for light-weight models.
* `_L_in` and `_L_out` specify the number of input and output features of the neural network, respectively. In this case, we will use 10 input features and 1 output feature, which are required for the `Diabetes` dataset.

Sometimes, very bad configurations appear in the initial design, leading to an unnecessarily long optimization process. An example is illustrated in @fig-divergence-threshold. To avoid this, we can set a divergence threshold in the `fun_control` dictionary. The `divergence_threshold` parameter is used to stop the optimization process if the objective function value exceeds a certain threshold. This is useful to avoid long optimization processes if the objective function is not well-behaved. We have set the `divergence_threshold=5_000`.

![TensorBoard visualization of the spotpython process.](figures_static/007_divergence_threshold.png){width="80%" #fig-divergence-threshold}


```{python}
fun_control = fun_control_init(
    PREFIX="S_3",
    max_time=1,
    data_set = Diabetes(),
    core_model_name="light.regression.NNLinearRegressor",
    hyperdict=LightHyperDict,
    divergence_threshold=5_000,
    _L_in=10,
    _L_out=1)

S_3 = Spot(fun=HyperLight().fun,
         fun_control=fun_control)
S_3.surrogate_control
```

```{python}
S_3.run()
S_3.print_results()
S_3.plot_progress(log_y=True)
```

Results can be printed using `Spot`'s `print_results()` method. The results include the best objective function value, the corresponding input values, and the number of function evaluations.

```{python}
results = S_3.print_results()
```

```{python}
S_3.surrogate_control
```

A formatted table of the results can be printed using the `print_res_table()` method:

```{python}
print_res_table(S_3)
```

The fitness landscape can be visualized using the `plot_important_hyperparameter_contour()` method:

```{python}
S_3.plot_important_hyperparameter_contour(max_imp=3)
```


## Organization

`Spot` organizes the surrogate based optimization process in four steps:

1. Selection of the objective function: `fun`.
2. Selection of the initial design: `design`.
3. Selection of the optimization algorithm: `optimizer`.
4. Selection of the surrogate model: `surrogate`.

For each of these steps, the user can specify an object:

```{python}
from spotpython.fun.objectivefunctions import Analytical
fun = Analytical().fun_sphere
from spotpython.design.spacefilling import SpaceFilling
design = SpaceFilling(2)
from scipy.optimize import differential_evolution
optimizer = differential_evolution
from spotpython.surrogate.kriging import Kriging
surrogate = Kriging()
```

For each of these steps, the user can specify a dictionary of control parameters.

1. `fun_control`
2. `design_control`
3. `optimizer_control`
4. `surrogate_control`

Each of these dictionaries has an initialization method, e.g., `fun_control_init()`. The initialization methods set the default values for the control parameters.

:::: {.callout-important}
#### Important:

* The specification of an lower bound in `fun_control` is mandatory.

:::

## Summary: Basic Attributes and Methods of the `Spot` Object

The `Spot` class organizes the surrogate based optimization process into the

* `fun` object with dictionry `fun_control`,
* `design` object with dictionary `design_control`,
* `optimizer` object with dictionary `optimizer_control`, and
* `surrogate` object with dictionary `surrogate_control`.

| Object | Description | Default |
| -- | -- | -- |
| `fun` | Can be one of the following: (1) any function from `spotpython.fun.*`, e.g., [fun_rosen](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/fun/objectivefunctions/#spotpython.fun.objectivefunctions.Analytical.fun_rosen), (2) a user-specified function (vectorized and accepting the `**kwargs` argument as, e.g., in @sec-user-specified-fun-branin), or, (3) if `PyTorch` hyperparameter tuning with `Lightning` is desired, the `HyperLight` class from `spotpython`, see [HyperLight](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/fun/hyperlight/). | There is no default, a simple example is `fun = Analytical().fun_sphere`|
| `design` | Can be any design generating function from `spotpython.design:*`, e.g., [Clustered](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/design/clustered/), [Factorial](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/design/factorial/), [Grid](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/design/factorial/), [Poor](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/design/poor/), [Random](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/design/random/), [Sobol](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/design/sobol/), or  [SpaceFilling](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/design/spacefilling/). | `design = SpaceFilling(k=15)` |
| `optimizer` | Can be any of `dual_annealing`, `direct`, `shgo`, `basinhopping`, or `differential_evolution`, see, e.g., [differential_evolution](hhttps://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution).| `optimizer=differential_evolution`|
| `surrogate` | Can be any `sklearn` regressor, e.g., [GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor) or, the [Kriging](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/surrogate/kriging/) class from `spotpython`.| `surrogate=Kriging()`|

| Dictionary | Code Reference | Example |
| -- | -- | -- |
| `fun_control` | [fun_control_init](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/utils/init/#spotpython.utils.init.fun_control_init) | The `lower` and  `upper` arguments are mandatory, e.g., `fun_control=fun_control_init(lower=np.array([-1, -1]),upper=np.array([1, 1]))`.|
| `design_control` | [design_control_init()](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/utils/init/#spotpython.utils.init.design_control_init) | `design_control=design_control_init()`|
| `optimizer_control` | [optimizer_control_init](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/utils/init/#spotpython.utils.init.optimizer_control_init) | `optimizer_control_init()`|
| `surrogate_control` |  [surrogate_control_init](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/utils/init/#spotpython.utils.init.surrogate_control_init) |`surrogate_control=surrogate_control_init()`


Based on the definition of the `fun`, `design`, `optimizer`, and `surrogate` objects, 
and their corresponding control parameter dictionaries, `fun_control`, `design_control`, `optimizer_control`, and `surrogate_control`,
the `Spot` object can be build as follows:


```{python}
fun_control=fun_control_init(lower=np.array([-1, -1]),
                            upper=np.array([1, 1]))
design_control=design_control_init()
optimizer_control=optimizer_control_init()
surrogate_control=surrogate_control_init()
spot_tuner = Spot(fun=fun,
                       fun_control=fun_control,
                       design_control=design_control,
                       optimizer_control=optimizer_control,
                       surrogate_control=surrogate_control)
```

## Run

```{python}
spot_tuner.run()
```

## Print the Results

```{python}
spot_tuner.print_results()
```

## Show the Progress

```{python}
spot_tuner.plot_progress()
```

## Visualize the Surrogate

* The plot method of the `kriging` surrogate is used.
* Note: the plot uses the interval defined by the ranges of the natural variables.

```{python}
spot_tuner.surrogate.plot()
```


## Init: Build Initial Design

```{python}
from spotpython.design.spacefilling import SpaceFilling
from spotpython.surrogate.kriging import Kriging
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.init import fun_control_init
gen = SpaceFilling(2)
rng = np.random.RandomState(1)
lower = np.array([-5,-0])
upper = np.array([10,15])
fun = Analytical().fun_branin

fun_control = fun_control_init(sigma=0)

X = gen.scipy_lhd(10, lower=lower, upper = upper)
print(X)
y = fun(X, fun_control=fun_control)
print(y)
```

## Replicability

Seed

```{python}
gen = SpaceFilling(2, seed=123)
X0 = gen.scipy_lhd(3)
gen = SpaceFilling(2, seed=345)
X1 = gen.scipy_lhd(3)
X2 = gen.scipy_lhd(3)
gen = SpaceFilling(2, seed=123)
X3 = gen.scipy_lhd(3)
X0, X1, X2, X3
```

## Surrogates

### A Simple Predictor

The code below shows how to use a simple model for prediction. Assume that only two (very costly) measurements are available:
  
  1. f(0) = 0.5
  2. f(2) = 2.5

We are interested in the value at $x_0 = 1$, i.e., $f(x_0 = 1)$, but cannot run an additional, third experiment.

```{python}
from sklearn import linear_model
X = np.array([[0], [2]])
y = np.array([0.5, 2.5])
S_lm = linear_model.LinearRegression()
S_lm = S_lm.fit(X, y)
X0 = np.array([[1]])
y0 = S_lm.predict(X0)
print(y0)
```

Central Idea: Evaluation of the surrogate model `S_lm` is much cheaper (or / and much faster) than running the real-world experiment $f$.


## Tensorboard Setup

### Tensorboard Configuration

The `TENSORBOARD_CLEAN` argument can be set to `True` in the `fun_control` dictionary to archive the TensorBoard folder if it already exists. This is useful if you want to start a hyperparameter tuning process from scratch. If you want to continue a hyperparameter tuning process, set `TENSORBOARD_CLEAN` to `False`. Then the TensorBoard folder will not be archived and the old and new TensorBoard files will shown in the TensorBoard dashboard.


### Starting TensorBoard {#sec-tensorboard-start}

`TensorBoard` can be started as a background process with the following command, where `./runs` is the default directory for the TensorBoard log files:

```{raw}
tensorboard --logdir="./runs"
```

::: {.callout-note}
#### TENSORBOARD_PATH

The TensorBoard path can be printed with the following command (after a `fun_control` object has been created):

```{python}
#| label: 024_tensorboard_path
#| eval: false
from spotpython.utils.init import get_tensorboard_path
get_tensorboard_path(fun_control)
```

:::


## Demo/Test: Objective Function Fails

SPOT expects `np.nan` values from failed objective function values. These are handled. Note: SPOT's counter considers only successful executions of the objective function.

```{python}
import numpy as np
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import Spot
import numpy as np
from math import inf
# number of initial points:
ni = 20
# number of points
n = 30

fun = Analytical().fun_random_error
fun_control=fun_control_init(
    lower = np.array([-1]),
    upper= np.array([1]),
    fun_evals = n,
    show_progress=False)
design_control=design_control_init(init_size=ni)

spot_1 = Spot(fun=fun,
                     fun_control=fun_control,
                     design_control=design_control)

# assert value error from the run method
try:
    spot_1.run()
except ValueError as e:
    print(e)
```

## Handling Results: Printing, Saving, and Loading

The results can be printed with the following command:

```{python}
#| label: a_04__print_results
#| eval: false
spot_tuner.print_results(print_screen=False)
```

The tuned hyperparameters can be obtained as a dictionary with the following command:

```{python}
#| label: a_04__get_tuned_hyperparameters
#| eval: false
from spotpython.hyperparameters.values import get_tuned_hyperparameters
get_tuned_hyperparameters(spot_tuner, fun_control)
```

The results can be saved and reloaded with the following commands:

```{python}
#| label: a_04__save_and_load
#| eval: false
from spotpython.utils.file import save_pickle, load_pickle
from spotpython.utils.init import get_experiment_name
experiment_name = get_experiment_name("024")
SAVE_AND_LOAD = False
if SAVE_AND_LOAD == True:
    save_pickle(spot_tuner, experiment_name)
    spot_tuner = load_pickle(experiment_name)
```

## `spotpython` as a Hyperparameter Tuner

### Modifying Hyperparameter Levels {#sec-modifying-hyperparameter-levels}


`spotpython` distinguishes between different types of hyperparameters. The following types are supported:

* `int` (integer)
* `float` (floating point number)
* `boolean` (boolean)
* `factor` (categorical)

#### Integer Hyperparameters

Integer hyperparameters can be modified with the `set_int_hyperparameter_values()` [[SOURCE]](https://sequential-parameter-optimization.github.io/spotpython/reference/spotpython/hyperparameters/values/#spotpython.hyperparameters.values.set_int_hyperparameter_values) function. The following code snippet shows how to modify the `n_estimators` hyperparameter of a random forest model:

```{python}
from spotriver.hyperdict.river_hyper_dict import RiverHyperDict
from spotpython.utils.init import fun_control_init
from spotpython.hyperparameters.values import set_int_hyperparameter_values
from spotpython.utils.eda import print_exp_table
fun_control = fun_control_init(
    core_model_name="forest.AMFRegressor",
    hyperdict=RiverHyperDict,
)
print("Before modification:")
print_exp_table(fun_control)
set_int_hyperparameter_values(fun_control, "n_estimators", 2, 5)
print("After modification:")
print_exp_table(fun_control)
```

#### Float Hyperparameters

Float hyperparameters can be modified with the `set_float_hyperparameter_values()` [[SOURCE]](https://sequential-parameter-optimization.github.io/spotpython/reference/spotpython/hyperparameters/values/#spotpython.hyperparameters.values.set_float_hyperparameter_values) function. The following code snippet shows how to modify the `step` hyperparameter of a hyperparameter of a Mondrian Regression Tree model:


```{python}
from spotriver.hyperdict.river_hyper_dict import RiverHyperDict
from spotpython.utils.init import fun_control_init
from spotpython.hyperparameters.values import set_float_hyperparameter_values
from spotpython.utils.eda import print_exp_table
fun_control = fun_control_init(
    core_model_name="forest.AMFRegressor",
    hyperdict=RiverHyperDict,
)
print("Before modification:")
print_exp_table(fun_control)
set_float_hyperparameter_values(fun_control, "step", 0.2, 5)
print("After modification:")
print_exp_table(fun_control)
```


#### Boolean Hyperparameters

Boolean hyperparameters can be modified with the `set_boolean_hyperparameter_values()` [[SOURCE]](https://sequential-parameter-optimization.github.io/spotpython/reference/spotpython/hyperparameters/values/#spotpython.hyperparameters.values.set_boolean_hyperparameter_values) function. The following code snippet shows how to modify the `use_aggregation` hyperparameter of a Mondrian Regression Tree model:

```{python}
from spotriver.hyperdict.river_hyper_dict import RiverHyperDict
from spotpython.utils.init import fun_control_init
from spotpython.hyperparameters.values import set_boolean_hyperparameter_values
from spotpython.utils.eda import print_exp_table
fun_control = fun_control_init(
    core_model_name="forest.AMFRegressor",
    hyperdict=RiverHyperDict,
)
print("Before modification:")
print_exp_table(fun_control)
set_boolean_hyperparameter_values(fun_control, "use_aggregation", 0, 0)
print("After modification:")
print_exp_table(fun_control)
```

#### Factor Hyperparameters

Factor hyperparameters can be modified with the `set_factor_hyperparameter_values()` [[SOURCE]](https://sequential-parameter-optimization.github.io/spotpython/reference/spotpython/hyperparameters/values/#spotpython.hyperparameters.values.set_factor_hyperparameter_values) function. The following code snippet shows how to modify the `leaf_model` hyperparameter of a Hoeffding Tree Regressor  model:

```{python}
from spotriver.hyperdict.river_hyper_dict import RiverHyperDict
from spotpython.utils.init import fun_control_init
from spotpython.hyperparameters.values import set_factor_hyperparameter_values
from spotpython.utils.eda import print_exp_table
fun_control = fun_control_init(
    core_model_name="tree.HoeffdingTreeRegressor",
    hyperdict=RiverHyperDict,
)
print("Before modification:")
print_exp_table(fun_control)
set_factor_hyperparameter_values(fun_control, "leaf_model", ['LinearRegression',
                                                    'Perceptron'])
print("After modification:")
```




## Sampling in spotpython

`spotpython` uses a class for generating space-filling designs using Latin Hypercube Sampling (LHS) and maximin distance criteria. It is based on `scipy`'s `LatinHypercube` class. The following example demonstrates how to generate a Latin Hypercube Sampling design using `spotpython`. The result is shown in @fig-lhs-spotpython. As can seen in the figure, a Latin hypercube sample generates $n$ points in $[0,1)^{d}$. Each univariate marginal distribution is stratified, placing exactly one point in $[j/n, (j+1)/n)$ for $j=0,1,...,n-1$.

```{python}
#| label: fig-lhs-spotpython
#| fig-cap: "Latin Hypercube Sampling design (sampling plan)"
import matplotlib.pyplot as plt
import numpy as np
from spotpython.design.spacefilling import SpaceFilling
lhd = SpaceFilling(k=2, seed=123)
X = lhd.scipy_lhd(n=10, repeats=1, lower=np.array([0, 0]), upper=np.array([10, 10]))
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()
```


## Example: `Spot` and the Sphere Function

Central Idea: Evaluation of the surrogate model `S` is much cheaper (or / and much faster) than running the real-world experiment $f$.
We start with a small example.


```{python}
import numpy as np
from math import inf
from spotpython.fun.objectivefunctions import Analytical
from spotpython.utils.init import fun_control_init, design_control_init
from spotpython.hyperparameters.values import set_control_key_value
from spotpython.spot import Spot
import matplotlib.pyplot as plt
```

### The Objective Function: Sphere

The `spotpython` package provides several classes of objective functions. We will use an analytical objective function, i.e., a function that can be described by a (closed) formula:
$$
f(x) = x^2
$$

```{python}
fun = Analytical().fun_sphere
```

We can apply the function `fun` to input values and plot the result:

```{python}
x = np.linspace(-1,1,100).reshape(-1,1)
y = fun(x)
plt.figure()
plt.plot(x, y, "k")
plt.show()
```

### The `Spot` Method as an Optimization Algorithm Using a Surrogate Model

We initialize the `fun_control` dictionary.
The `fun_control` dictionary contains the parameters for the objective function.
The `fun_control` dictionary is passed to the `Spot` method.

```{python}
fun_control=fun_control_init(lower = np.array([-1]),
                     upper = np.array([1]))
spot_0 = Spot(fun=fun,
                   fun_control=fun_control)
spot_0.run()
```

The method `print_results()` prints the results, i.e., the best objective function value ("min y") and the corresponding input value ("x0").

```{python}
spot_0.print_results()
```

To plot the search progress, the method `plot_progress()` can be used. The parameter `log_y` is used to plot the objective function values on a logarithmic scale.

```{python}
#| label: fig-spot-progress
#| fig-cap: "Visualization of the search progress of the `Spot` method. The black elements (points and line) represent the initial design, before the surrogate is build. The red elements represent the search on the surrogate."
spot_0.plot_progress(log_y=True)
```

If the dimension of the input space is one, the method `plot_model()` can be used to visualize the model and the underlying objective function values.

```{python}
#| label: fig-spot-model-1d
#| fig-cap: "Visualization of the model and the underlying objective function values."
spot_0.plot_model()
```

## `Spot` Parameters: `fun_evals`, `init_size` and `show_models`

We will modify three parameters:

  1. The number of function evaluations (`fun_evals`) will be set to `10` (instead of 15, which is the default value) in the `fun_control` dictionary.
  2. The parameter `show_models`, which visualizes the search process for each single iteration for 1-dim functions, in the `fun_control` dictionary.
  3. The size of the initial design (`init_size`) in the `design_control` dictionary.


The full list of the `Spot` parameters is shown in code reference on GitHub, see [Spot](https://sequential-parameter-optimization.github.io/spotpython/reference/spotpython/spot/spot/#spotpython.Spot).

```{python}
fun_control=fun_control_init(lower = np.array([-1]),
                     upper = np.array([1]),
                     fun_evals = 10,
                     show_models = True)               
design_control = design_control_init(init_size=9)
spot_1 = Spot(fun=fun,
                   fun_control=fun_control,
                   design_control=design_control)
spot_1.run()
```

## Print the Results

```{python}
spot_1.print_results()
```

## Show the Progress

```{python}
spot_1.plot_progress()
```


## Visualizing the Optimization and Hyperparameter Tuning Process with TensorBoard {#sec-visualizing-tensorboard-01}

`spotpython` supports the visualization of the hyperparameter tuning process with TensorBoard. The following example shows how to use TensorBoard with `spotpython`.

First, we define an "PREFIX" to identify the hyperparameter tuning process. The PREFIX is used to create a directory for the TensorBoard files.

```{python}
#| label: code-spot-tensorboard
fun_control = fun_control_init(
    PREFIX = "01",
    lower = np.array([-1]),
    upper = np.array([2]),
    fun_evals=100,
    TENSORBOARD_CLEAN=True,
    tensorboard_log=True)
design_control = design_control_init(init_size=5)
```

Since the `tensorboard_log` is `True`, `spotpython` will log the optimization process in the TensorBoard files.
The argument `TENSORBOARD_CLEAN=True` will move the TensorBoard files from the previous run to a backup folder, so that  TensorBoard files from previous runs are not overwritten and a clean start in the `runs` folder is guaranteed.

```{python}
spot_tuner = Spot(fun=fun,                   
                   fun_control=fun_control,
                   design_control=design_control)
spot_tuner.run()
spot_tuner.print_results()
```


Now we can start TensorBoard in the background. The TensorBoard process will read the TensorBoard files and visualize the hyperparameter tuning process.
From the terminal, we can start TensorBoard with the following command:

```{raw}
tensorboard --logdir="./runs"
```

`logdir` is the directory where the TensorBoard files are stored. In our case, the TensorBoard files are stored in the directory `./runs`.

TensorBoard will start a web server on port 6006. We can access the TensorBoard web server with the following URL:

```{raw}
http://localhost:6006/
```

The first TensorBoard visualization shows the objective function values plotted against the wall time. The wall time is the time that has passed since the start of the hyperparameter tuning process. The five initial design points are shown in the upper left region of the plot. The line visualizes the optimization process.
![TensorBoard visualization of the spotpython process. Objective function values plotted against wall time.](figures_static/01_tensorboard_01.png)

The second TensorBoard visualization shows the input values, i.e., $x_0$, plotted against the wall time.
![TensorBoard visualization of the spotpython process.](figures_static/01_tensorboard_02.png)

The third TensorBoard plot illustrates how `spotpython` can be used as a microscope for the internal mechanisms of the surrogate-based optimization process. Here, one important parameter, the learning rate $\theta$ of the Kriging surrogate is plotted against the number of optimization steps.

![TensorBoard visualization of the spotpython process.](figures_static/01_tensorboard_03.png){width="50%"}


## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this lecture is available on GitHub in the [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/007_spot_intro.ipynb)

:::
