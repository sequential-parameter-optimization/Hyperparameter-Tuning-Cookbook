# Preface  {.unnumbered}

> This document provides a comprehensive guide to hyperparameter tuning using spotpython for scikit-learn, scipy-optimize, River, and PyTorch. The first part introduces fundamental ideas from optimization. The second part discusses numerical issues and introduces spotpython's surrogate model-based optimization process. The thirs part focuses on hyperparameter tuning. Several case studies are presented, including hyperparameter tuning for sklearn models such as Support Vector Classification, Random Forests, Gradient Boosting (XGB), and K-nearest neighbors (KNN), as well as a Hoeffding Adaptive Tree Regressor from river. The integration of spotpython into the PyTorch and PyTorch Lightning training workflow is also discussed. With a hands-on approach and step-by-step explanations, this cookbook serves as a practical starting point for anyone interested in hyperparameter tuning with Python. Highlights include the interplay between Tensorboard, PyTorch Lightning, spotpython, spotriver, and River. This publication is under development, with updates available on the corresponding webpage.


:::{.callout-important}
## Important: This book is still under development.
The most recent version of this book is available at [https://sequential-parameter-optimization.github.io/Hyperparameter-Tuning-Cookbook/](https://sequential-parameter-optimization.github.io/Hyperparameter-Tuning-Cookbook/)
:::

## Book Structure {.unnunmbered}

This document is structured in three parts. The first part presents an introduction to optimization. The second part describes
numerical methods, and the third part presents hyperparameter tuning.


::: {.callout-tip}
#### Hyperparameter Tuning Reference
* The open access book @bart21i provides a comprehensive overview of hyperparameter tuning. It can be downloaded from [https://link.springer.com/book/10.1007/978-981-19-5170-1](https://link.springer.com/book/10.1007/978-981-19-5170-1).
:::

::: {.callout-note}
The ` .ipynb` notebook [@bart23e] is updated regularly and reflects updates and changes in the `spotpython` package.
It can be downloaded from [https://github.com/sequential-parameter-optimization/spotpython/blob/main/notebooks/14_spot_ray_hpt_torch_cifar10.ipynb](https://github.com/sequential-parameter-optimization/spotpython/blob/main/notebooks/14_spot_ray_hpt_torch_cifar10.ipynb).
:::

## Software Used in this Book {.unnumbered}

[scikit-learn](https://scikit-learn.org) is a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license. The project was started in 2007 by David Cournapeau as a Google Summer of Code project, and since then many volunteers have contributed.

[PyTorch](https://pytorch.org) is an optimized tensor library for deep learning using GPUs and CPUs. [Lightning](https://lightning.ai/docs/pytorch/latest/) is a lightweight PyTorch wrapper for high-performance AI research. It allows you to decouple the research from the engineering.

[River](https://riverml.xyz) is a Python library for online machine learning. It is designed to be used in real-world environments, where not all data is available at once, but streaming in.

[spotpython](https://github.com/sequential-parameter-optimization/spotpython) ("Sequential Parameter Optimization Toolbox in Python") is the Python version of the well-known hyperparameter tuner SPOT, which has been developed in the R programming environment for statistical analysis for over a decade. The related open-access book is available here: [Hyperparameter Tuning for Machine and Deep Learning with R---A Practical Guide](https://link.springer.com/book/10.1007/978-981-19-5170-1).


[spotriver](https://github.com/sequential-parameter-optimization/spotriver) provides an interface between [spotpython](https://github.com/sequential-parameter-optimization/spotpython) and [River](https://riverml.xyz).


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
        pages = {arXiv:2307.10262},
          doi = {10.48550/arXiv.2307.10262},
archivePrefix = {arXiv},
       eprint = {2307.10262},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230710262B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


```
