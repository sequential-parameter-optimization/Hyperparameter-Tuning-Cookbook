---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
jupyter: python3
---

# Introduction to Sequential Parameter Optimization


```{python}
import numpy as np
from math import inf
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import Spot
from scipy.optimize import shgo
from scipy.optimize import direct
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
```

This document describes the `Spot` features. The official `spotpython` documentation can be found here: [https://sequential-parameter-optimization.github.io/spotpython/](https://sequential-parameter-optimization.github.io/spotpython/).

## An Initial Example

The `spotpython` package provides several classes of objective functions. We will use an analytical objective function, i.e., a function that can be described by a (closed) formula:
$$
f(x) = x^2.
$$

```{python}
fun = Analytical().fun_sphere
```

```{python}
x = np.linspace(-1,1,100).reshape(-1,1)
y = fun(x)
plt.figure()
plt.plot(x,y, "k")
plt.show()
```

```{python}
from spotpython.utils.init import fun_control_init, design_control_init, surrogate_control_init, optimizer_control_init
spot_1 = Spot(fun=fun,
                   fun_control=fun_control_init(
                        lower = np.array([-10]),
                        upper = np.array([100]),
                        fun_evals = 7,
                        fun_repeats = 1,
                        max_time = inf,
                        noise = False,
                        tolerance_x = np.sqrt(np.spacing(1)),
                        var_type=["num"],
                        infill_criterion = "y",
                        n_points = 1,
                        seed=123,
                        log_level = 50),
                   design_control=design_control_init(
                        init_size=5,
                        repeats=1),
                   surrogate_control=surrogate_control_init(
                        method="interpolation",
                        min_theta=-4,
                        max_theta=3,
                        n_theta=1,
                        model_optimizer=differential_evolution,
                        model_fun_evals=10000))
spot_1.run()
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

Each of these dictionaries has an initialzaion method, e.g., `fun_control_init()`. The initialization methods set the default values for the control parameters.

:::: {.callout-important}
#### Important:

* The specification of an lower bound in `fun_control` is mandatory.

::: 

```{python}
from spotpython.utils.init import fun_control_init, design_control_init, optimizer_control_init, surrogate_control_init
fun_control=fun_control_init(lower=np.array([-1, -1]),
                            upper=np.array([1, 1]))
design_control=design_control_init()
optimizer_control=optimizer_control_init()
surrogate_control=surrogate_control_init()
```

## The Spot Object

Based on the definition of the `fun`, `design`, `optimizer`, and `surrogate` objects, 
and their corresponding control parameter dictionaries, `fun_control`, `design_control`, `optimizer_control`, and `surrogate_control`,
the `spot` object can be build as follows:

```{python}
from spotpython.spot import Spot
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

## Run With a Specific Start Design

To pass a specific start design, use the `X_start` argument of the `run` method.

```{python}
spot_x0 = Spot(fun=fun,
                    fun_control=fun_control_init(
                        lower = np.array([-10]),
                        upper = np.array([100]),
                        fun_evals = 7,
                        fun_repeats = 1,
                        max_time = inf,
                        noise = False,
                        tolerance_x = np.sqrt(np.spacing(1)),
                        var_type=["num"],
                        infill_criterion = "y",
                        n_points = 1,
                        seed=123,
                        log_level = 50),
                    design_control=design_control_init(
                        init_size=5,
                        repeats=1),
                    surrogate_control=surrogate_control_init(
                        method="interpolation",
                        min_theta=-4,
                        max_theta=3,
                        n_theta=1,
                        model_optimizer=differential_evolution,
                        model_fun_evals=10000))
spot_x0.run(X_start=np.array([0.5, -0.5]))
spot_x0.plot_progress()
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


# Introduction to spotpython {#sec-spot}

Surrogate model based optimization methods are common approaches in simulation and optimization. SPOT was developed because there is a great need for sound statistical analysis of simulation and optimization algorithms. SPOT includes methods for tuning based on classical regression and analysis of variance techniques.
It presents tree-based models such as classification and regression trees and random forests as well as Bayesian optimization (Gaussian process models, also known as Kriging). Combinations of different meta-modeling approaches are possible. SPOT comes with a sophisticated surrogate model based optimization method, that can handle discrete and continuous inputs. Furthermore, any model implemented in `scikit-learn` can be used out-of-the-box as a surrogate in `spotpython`.

SPOT implements key techniques such as exploratory fitness landscape analysis and sensitivity analysis. It can be used to understand the performance of various algorithms, while simultaneously giving insights into their algorithmic behavior.

The `spot` loop consists of the following steps:

1. Init: Build initial design $X$
2. Evaluate initial design on real objective $f$: $y = f(X)$
3. Build surrogate: $S = S(X,y)$
4. Optimize on surrogate: $X_0 =  \text{optimize}(S)$
5. Evaluate on real objective: $y_0 = f(X_0)$
6. Impute (Infill) new points: $X = X \cup X_0$, $y = y \cup y_0$.
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


## Advantages of the spotpython approach

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
* Flexible, modular meta-modeling handling. spotpython come with a Kriging model, which can be replaced by any model implemented in `scikit-learn`.
* Enhanced metric handling, especially for categorical hyperparameters (any sklearn metric can be used). Default is..
* Integration with TensorBoard: Visualization of the hyperparameter tuning process, of the training steps, the model graph. Parallel coordinates plot, scatter plot matrix, and more.
* Reproducibility. Results are stored as pickle files. The results can be loaded and visualized at any time and be transferred between different machines and operating systems.
* Handles scikit-learn models and pytorch models out-of-the-box. The user has to add a simple wrapper for passing the hyperparemeters to use a pytorch model in spotpython.
* Compatible with Lightning.
* User can add own models as plain python code.
* User can add own data sets in various formats.
* Flexible data handling and data preprocessing.
* Many examples online (hyperparameter-tuning-cookbook).
* spotpython uses a robust optimizer that can even deal with hyperparameter-settings that cause crashes of the algorithms to be tuned.
* even if the optimum is not found, HPT with spotpython prevents the user from choosing bad hyperparameters in a systematic way (design of experiments).


## Disadvantages of the spotpython approach

* Time consuming
* Surrogate can be misguiding
* no parallelization implement yet



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
