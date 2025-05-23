---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
---

# Optimal Computational Budget Allocation in spotpython {#sec-ocba}

This chapter demonstrates how noisy functions can be handled `spotpython`:

* First, Optimal Computational Budget Allocation (OCBA) is introduced in @sec-ocba.
* Then, the nugget effect is explained in @sec-nugget.



## Citation {.unnumbered}

If this document has been useful to you and you wish to cite it in a scientific publication, please refer to the following paper, which can be found on arXiv: [https://arxiv.org/abs/2307.10262](https://arxiv.org/abs/2307.10262).


```{bibtex}
@ARTICLE{bart23iArXiv,
      author = {{Bartz-Beielstein}, Thomas},
      title = "{Hyperparameter Tuning Cookbook:
          A guide for scikit-learn, PyTorch, river, and spotpython}",
     journal = {arXiv e-prints},
    keywords = {Computer Science - Machine Learning,
      Computer Science - Artificial Intelligence, 90C26, I.2.6, G.1.6},
         year = 2023,
        month = jul,
          eid = {arXiv:2307.10262},
          doi = {10.48550/arXiv.2307.10262},
archivePrefix = {arXiv},
       eprint = {2307.10262},
 primaryClass = {cs.LG}
}
```




## Example: `spotpython`, OCBA,  and the Noisy Sphere Function

```{python}
import numpy as np
from math import inf
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import Spot
import matplotlib.pyplot as plt
from spotpython.utils.init import fun_control_init, get_spot_tensorboard_path
from spotpython.utils.init import fun_control_init, design_control_init, surrogate_control_init

PREFIX = "14"

```

### The Objective Function: Noisy Sphere {#sec-noisy-sphere}

The `spotpython` package provides several classes of objective functions. We will use an analytical objective function with noise, i.e., a function that can be described by a (closed) formula:
   $$f(x) = x^2 + \epsilon$$

Since `sigma` is set to `0.1`, noise is added to the function:

```{python}
fun = Analytical().fun_sphere
fun_control = fun_control_init(
    PREFIX=PREFIX,
    sigma=0.1)
```

A plot (@fig-noisy-sphere-14) illustrates the noise:

```{python}
#| label: fig-noisy-sphere-14
#| fig-cap: "The noisy sphere function with noise."
x = np.linspace(-1,1,100).reshape(-1,1)
y = fun(x, fun_control=fun_control)
plt.figure()
plt.plot(x,y, "k")
plt.show()
```

::: {.callout-note}

#### Noise Handling in `spotpython`

`spotpython` has several options to cope with noisy functions:

  1. `fun_repeats` is set to a value larger than 1, e.g., 2, which means every function evaluation during the search on the surrogate is repeated twice. The mean of the two evaluations is used as the function value.
  2. `init size` (of the `design_control` dictionary) is set to a value larger than 1 (here: 2).
  3. `ocba_delta` is set to a value larger than 1 (here: 2). This means that the OCBA algorithm is used to allocate the computational budget optimally.
  4. Using a nugget term in the surrogate model. This is done by setting `method="regression"` in the `surrogate_control` dictionary. An example is given in @sec-nugget.

::: 

## Using Optimal Computational Budget Allocation (OCBA) {#sec-ocba-example}

The [Optimal Computational Budget Allocation](https://en.wikipedia.org/wiki/Optimal_computing_budget_allocation)  (OCBA) algorithm is a powerful tool for efficiently distributing computational resources [@Chen10a]. It is specifically designed to maximize the Probability of Correct Selection (PCS) while minimizing computational costs. By strategically allocating more simulation effort to design alternatives that are either more challenging to evaluate or more likely to yield optimal results, OCBA ensures an efficient use of resources. This approach enables researchers and decision-makers to achieve accurate outcomes more quickly and with fewer computational demands, making it an invaluable method for simulation optimization.


The OCBA algorithm is implemented in `spotpython` and can be used by setting `ocba_delta` to a value larger than `0`. The source code is available in the `spotpython` package, see [[DOC]](https://sequential-parameter-optimization.github.io/spotPython/reference/spotpython/budget/ocba/).
See also @Bart11b.

::: {#exm-ocba}
To reproduce the example from p.49 in @Chen10a, the following `spotpython` code can be used:
```{python}
import numpy as np
from spotpython.budget.ocba import get_ocba
mean_y = np.array([1,2,3,4,5])
var_y = np.array([1,1,9,9,4])
get_ocba(mean_y, var_y, 50)
```
::: 


### The Noisy Sphere

We will demonstrate the OCBA algorithm on the noisy sphere function defined in @sec-noisy-sphere. The OCBA algorithm is used to allocate the computational budget optimally. This means that the function evaluations are repeated several times, and the best function value is used for the next iteration.

::: {.callout-note}
#### Visualizing the Search of the OCBA Algorithm

* The `show_models` parameter in the `fun_control` dictionary is set to `True`. This means that the surrogate model is shown during the search.
* To keep the visualization simple, only the ground truth and the surrogate model are shown. The surrogate model is shown in blue, and the ground truth is shown in orange. The noisy function was shown in @fig-noisy-sphere-14.

::: 

```{python}
spot_1_noisy = Spot(fun=fun,
                   fun_control=fun_control_init( 
                   lower = np.array([-1]),
                   upper = np.array([1]),
                   fun_evals = 20,
                   fun_repeats = 1,
                   noise = True,
                   tolerance_x=0.0,
                   ocba_delta = 2,                   
                   show_models=True),
                   design_control=design_control_init(init_size=5, repeats=2),
                   surrogate_control=surrogate_control_init(method="regression"))
```

```{python}
spot_1_noisy.run()
```

### Print the Results

```{python}
spot_1_noisy.print_results()
```

```{python}
spot_1_noisy.plot_progress(log_y=True)
```

## Noise and Surrogates: The Nugget Effect {#sec-nugget}

In the previous example, we have seen that the `fun_repeats` parameter can be used to repeat function evaluations. This is useful when the function is noisy. However, it is not always possible to repeat function evaluations, e.g., when the function is expensive to evaluate.
In this case, we can use a surrogate model with a nugget term. The nugget term is a small value that is added to the diagonal of the covariance matrix. This allows the surrogate model to fit the data better, even if the data is noisy. The nugget term is added, if `method="regression"` is set in the `surrogate_control` dictionary.

### The Noisy Sphere

#### The Data

We prepare some data first:

```{python}
import numpy as np
import spotpython
from spotpython.fun.objectivefunctions import Analytical
from spotpython.spot import Spot
from spotpython.design.spacefilling import SpaceFilling
from spotpython.surrogate.kriging import Kriging
import matplotlib.pyplot as plt

gen = SpaceFilling(1)
rng = np.random.RandomState(1)
lower = np.array([-10])
upper = np.array([10])
fun = Analytical().fun_sphere
fun_control = fun_control_init(    
    sigma=2,
    seed=125)
X = gen.scipy_lhd(10, lower=lower, upper = upper)
y = fun(X, fun_control=fun_control)
X_train = X.reshape(-1,1)
y_train = y
```

A surrogate without nugget is fitted to these data:

```{python}
S = Kriging(name='kriging',
            seed=123,
            log_level=50,
            n_theta=1,
            method="interpolation")
S.fit(X_train, y_train)

X_axis = np.linspace(start=-13, stop=13, num=1000).reshape(-1, 1)
mean_prediction, std_prediction, ei = S.predict(X_axis, return_val="all")

plt.scatter(X_train, y_train, label="Observations")
plt.plot(X_axis, mean_prediction, label="mue")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Sphere: Gaussian process regression on noisy dataset")
```

In comparison to the surrogate without nugget, we fit a surrogate with nugget to the data:

```{python}
S_nug = Kriging(name='kriging',
            seed=123,
            log_level=50,
            n_theta=1,
            method="regression")
S_nug.fit(X_train, y_train)
X_axis = np.linspace(start=-13, stop=13, num=1000).reshape(-1, 1)
mean_prediction, std_prediction, ei = S_nug.predict(X_axis, return_val="all")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X_axis, mean_prediction, label="mue")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Sphere: Gaussian process regression with nugget on noisy dataset")
```

The value of the nugget term can be extracted from the model as follows:

```{python}
S.Lambda
```

```{python}
10**S_nug.Lambda
```

We see:

  * the first model `S` has no nugget, 
  * whereas the second model has a nugget value (`Lambda`) larger than zero.

## Exercises

### Noisy `fun_cubed`

Analyse the effect of noise on the `fun_cubed` function with the following settings:

```{python}
fun = Analytical().fun_cubed
fun_control = fun_control_init(    
    sigma=10,
    seed=123)
lower = np.array([-10])
upper = np.array([10])
```

### `fun_runge`

Analyse the effect of noise on the `fun_runge` function with the following settings:

```{python}
lower = np.array([-10])
upper = np.array([10])
fun = Analytical().fun_runge
fun_control = fun_control_init(    
    sigma=0.25,
    seed=123)
```


### `fun_forrester`

Analyse the effect of noise on the `fun_forrester` function with the following settings: 

```{python}
lower = np.array([0])
upper = np.array([1])
fun = Analytical().fun_forrester
fun_control = {"sigma": 5,
               "seed": 123}
```


### `fun_xsin`

Analyse the effect of noise on the `fun_xsin` function with the following settings: 

```{python}
lower = np.array([-1.])
upper = np.array([1.])
fun = Analytical().fun_xsin
fun_control = fun_control_init(    
    sigma=0.5,
    seed=123)
```




## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this chapter is available on GitHub in the [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/014_num_spot_ocba.ipynb)

:::
