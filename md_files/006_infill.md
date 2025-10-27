---
execute:
  cache: false
  eval: true
  echo: true
  warning: false
---

# Infill Criteria

In the context of computer experiments and surrogate modeling, a sampling plan refers to the set of input values, often denoted as $X$, at which a computer code is evaluated. The primary objective of a sampling plan is to efficiently explore the input space to understand the behavior of the computer code and to construct a surrogate model that accurately represents that behavior. Historically, Response Surface Methodology (RSM) provided methods for designing such plans, often based on rectangular grids or factorial designs. More recently, Design and Analysis of Computer Experiments (DACE) has emerged as a more flexible and powerful approach for this purpose.

A surrogate model, or $\hat{f}$, is built to approximate the expensive response of a black-box function $f(x)$. Since evaluating $f$ is costly, only a sparse set of samples is used to construct $\hat{f}$, which can then provide inexpensive predictions for any point in the design space. However, as a surrogate model is inherently an approximation of the true function, its accuracy and predictive capabilities can be significantly improved by incorporating new data points, known as infill points. Infill points are strategically chosen to either reduce uncertainty, improve predictions in specific regions of interest, or enhance the model's ability to identify optima or trends.

The process of updating a surrogate model with infill points is iterative. It typically involves:

*   **Identifying Regions of Interest**: Analyzing the current surrogate model to determine areas where it is inaccurate, has high uncertainty, or predicts promising results (e.g., potential optima).
*   **Selecting Infill Points**: Choosing new data points based on specific criteria that balance different objectives.
*   **Evaluating the True Function**: Running the actual simulation or experiment at the selected infill points to obtain their corresponding outputs.
*   **Updating the Surrogate Model**: Retraining or updating the surrogate model using the new, augmented dataset.
*   **Repeating**: Iterating this process until the model meets predefined accuracy criteria or the computational budget is exhausted.

## Balancing Exploitation and Exploration

A crucial aspect of selecting infill points is navigating the inherent trade-off between exploitation and exploration.

::: {#def-exploitation}
#### Exploitation 

Exploitation refers to sampling near predicted optima to refine the solution. This strategy aims to rapidly converge on a good solution by focusing computational effort where the surrogate model suggests the best values might lie.
:::

::: {#def-exploration}
#### Exploration
Exploration involves sampling in regions of high uncertainty to improve the global accuracy of the model. This approach ensures that the model is well-informed across the entire design space, preventing it from getting stuck in local optima.
:::

@Forr08a emphasizes that effective infill criteria are designed to combine both exploitation and exploration.

## Expected Improvement (EI)

Expected Improvement (EI) is one of the most influential and widely-used infill criteria.
Formalized by @Jones1998 and building upon the work of @mockus1978toward, EI provides a mathematically elegant framework that naturally balances exploitation and exploration. Rather than simply picking the point with the best predicted value (pure exploitation) or the point with the highest uncertainty (pure exploration), EI asks a more nuanced question: "How much improvement over the current best solution can we *expect* to gain by evaluating the true function at a new point $x$?".

The Expected Improvement, $EI(x)$, can be calculated using the following formula:

$$
EI(x) = \sigma(x) \left[ Z \Phi(Z) + \phi(Z) \right]
$$
where:

*   $\mu(x)$ (or $\hat{y}(x)$) is the Kriging prediction (mean of the stochastic process) at a new, unobserved point $x$.
*   $\sigma(x)$ (or $\hat{s}(x)$) is the estimated standard deviation (square root of the variance $\hat{s}^2(x)$) of the prediction at point $x$.
*   $f_{best}$ (or $y_{min}$) is the best (minimum, for minimization problems) observed function value found so far.
*   $Z = \frac{f_{best} - \mu(x)}{\sigma(x)}$ is the standardized improvement.
*   $\Phi(Z)$ is the cumulative distribution function (CDF) of the standard normal distribution.
*   $\phi(Z)$ is the probability density function (PDF) of the standard normal distribution.

If $\sigma(x) = 0$ (meaning there is no uncertainty at point $x$, typically because it's an already sampled point), then $EI(x) = 0$, reflecting the intuition that no further improvement can be expected at a known point. A maximization of Expected Improvement as an infill criterion will eventually lead to the global optimum.

The elegance of the EI formula lies in its combination of two distinct terms:

*   **Exploitation Term**: $(f_{best} - \mu(x)) \Phi(Z)$. This part of the formula contributes more when the predicted value $\mu(x)$ is significantly lower (better) than the current best observed value $f_{best}$. It is weighted by the probability $\Phi(Z)$ that the true function value at $x$ will indeed be an improvement over $f_{best}$.
*   **Exploration Term**: $\sigma(x) \phi(Z)$. This term becomes larger when there is high uncertainty ($\sigma(x)$ is large) in the model's prediction at $x$. It accounts for the potential of discovering unexpectedly good values in areas that have not been thoroughly explored, even if the current mean prediction there is not the absolute best.

Expected Improvement offers several significant practical benefits:

*   **Automatic Balance**: It inherently balances exploitation and exploration without requiring any manual adjustment of weights or parameters.
*   **Scale Invariance**: EI is relatively insensitive to the scaling of the objective function, making it robust across various problem types.
*   **Theoretical Foundation**: It is underpinned by a strong theoretical basis derived from decision theory and information theory.
*   **Efficient Optimization**: The smooth and differentiable nature of the EI function allows for efficient optimization using gradient-based algorithms to find the next infill point.
*   **Proven Performance**: EI has demonstrated consistent and strong performance in numerous real-world applications across various domains.

## Expected Improvement in the Hyperparameter Tuning Cookbook (Python Implementation)

Within the context of the Hyperparameter Tuning Cookbook, Expected Improvement serves a critical role in Sequential Model-Based Optimization. It systematically guides the selection of which hyperparameter configurations to evaluate next, facilitating the efficient utilization of computational resources. By intelligently balancing the need to exploit promising regions and explore uncertain areas, EI helps identify optimal hyperparameters with a reduced number of expensive model training runs. This provides a principled and automated method for navigating complex hyperparameter spaces without extensive manual intervention.

While the foundational concepts in @Forr08a are often illustrated with MATLAB code, the Hyperparameter Tuning Cookbook emphasizes and provides implementations in Python. The `spotpython` package, consistent with the Cookbook's approach, provides a Python implementation of Expected Improvement within its Kriging class. For minimization problems, `spotpython` typically calculates and returns the negative Expected Improvement, aligning with standard optimization algorithm conventions. Furthermore, to enhance numerical stability and mitigate issues when EI values are very small, `spotpython` often works with a logarithmic transformation of EI and incorporates a small epsilon value.



## Jupyter Notebook

:::{.callout-note}

* The Jupyter-Notebook of this lecture is available on GitHub in the [Hyperparameter-Tuning-Cookbook Repository](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/blob/main/006_infill.ipynb)

:::


