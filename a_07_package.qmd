---
execute:
  cache: false
  eval: false
  echo: true
  warning: false
title: Python Package Building
jupyter: python3
---

## Introduction {.unnumbered}

This notebook will guide you through the process of creating a Python package.

* All examples can be found in the `userPackage` directory, see: [userPackage](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/tree/main/userPackage)

## Create a Conda Environment

* `conda create -n userpackage python=3.12`

* `conda activate userpackage`

* Install the following packages:

```bash
python -m pip install build flake8 black mkdocs mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index mkdocs-material mkdocs-exclude mkdocstrings mkdocstrings-python tensorflow twine jupyter matplotlib plotly pandas pytest spotpython
```

## Download the User Package

The user package can be found in the `userPackage` directory, see: [userPackage](https://github.com/sequential-parameter-optimization/Hyperparameter-Tuning-Cookbook/tree/main/userPackage)

## Build the User Package

* cd into the `userPackage` directory and run the following command:
  * `./makefile.sh`
  * Alternatively, you can run the following commands:
      * `rm -f dist/userpackage*; python -m build; python -m pip install dist/userpackage*.tar.gz`    
      * `python -m mkdocs build`

## Open the Documentation of the User Package

* `mkdocs serve` to view the documentation


