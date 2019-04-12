# Statistical Inference of Discretely Observed Compound Poisson Processes and Related Jump Processes

Code for reproducing the simulations from my Part III Essay.

## This folder contains

#### **`demos`**

Folder containing four subfolders, one for each estimator in the essay.

* `spectralconv`
  - `spectralconvdemo.py` --- runs spectralconv estimator and plots against true density
  - `plots` --- output plots

* `spectralkde`
  - `spectralkdedemo.py` --- runs spectralkde estimator and plots against true density
  - `plots` --- output plots

* `bayesianmixture`
  - `bayesianmixturedemo.py` --- runs bayesianmixture estimator and plots against true density
  - `plots` --- output plots

* `bayesiandirichlet`
  - `bayesiandirichlet.py` --- runs bayesiandirichlet estimator and plots against true density
  - `plots` --- output plots

#### **`utils`**

Folder with utility classes and functions.

* `distributions.py`--- Distributions used during the simulations 

* `simulation.py`--- Simulates the CPP and observes points at discrete points in time

* `charfunctions.py`--- Characteristic functions of kernels/other functions required for the estimators

* `spectralconv.py`--- Kernel density estimator via computing estimators of convolution powers

* `spectralkde.py`--- Kernel density estimator via suitable inversion of characteristic functions

* `bayesianmixture.py`--- Parametric Bayesian density estimator via data augmentation scheme 

* `bayesiandirichlet.py`--- Non-parametric Bayesian density estimator via Dirichlet Process Mixture prior

