# Full GRF Estimation

This repository contains the source code implemented for my thesis "**Estimating Ground Reaction Force Based on Pressure and Inertia Measured by Smart Insoles** - A Deep Learning Approach"
I compared three multi-output regression methods for making these estimations:
1. single-target method
2. stacking
3. multi-task learning with hard parameter sharing

A multilayer perceptron (MLP) forms the basis for all of these methods.
The implementation of all devised models can  be found in `./models/`.

Two of these methods demonstrated merit, namely 1 and 3.
The code for training, hyperparameter tuning and cross-validation can be found, respectively, in:
- ``./1_single_target_method/``
- ``./3_multi-task_method/``

These implementations have quite similar pipelines.
The commonalities between both have been extracted in `data_processing.py` and `model_evaluation.py`.

The second, stacking method did not live up to its expectations.
To avoid confusion (and a lot of thorough refactoring), the hyperparameter tuning and model training of this method have been excluded from this repository.
To still give some insight into how it was done and why we decided to omit it, we added ``./experimentation/``.
This contains a jupyter notebook contains a simplified version of the experimentation made during this thesis.
It was made for the course "Statistical Foundations of Machine Learning".


## Getting Started

The `train.py` and `test.py` scripts have been added to easily reproduce the results from my thesis.
To get started, clone this repo and open the `grf-estimation/` directory in terminal.
From there, running the `make` command will install all necessary requirements, and train and test the best performing inter-subject model.
The model and test results will be saved in the `results/` directory.

To train and test different models, you can run the `train.py` and `test.py` scripts with different parameters.
To list all possible parameters use `--help` (e.g. `python train.py --help`).