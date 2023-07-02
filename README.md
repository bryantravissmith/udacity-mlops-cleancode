- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Proect Description
This is the project for the Udacity ML-Ops Nano degree.  The purpose is to
convert an ipython notebook to a library/model that is tested.

The original notebook uses python 3.8, so the setup and test are structure to
use the original environemnt for the notebook.

## Files and data description
There are the following files:

`churn_notebook.ipynb` - the originial notebook
`churn_libray.py` - libary to reproduces the originial model
`churn_script_logging_and_test.py` - the tests for the library

The data is stored as a csv in the `data/` directory.   The outputs of the code
are in each directory:

`images/` - EDA and Model Performance plots
`modesl/` - The best fit models
`logs/` - logs for the tests

## Running Files
This code is run with docker containers to reporduce the original environement.

Use the following comments to start jupyter labs to run the notebook and
interactively run the code

```
docker build -t mlops-cleancode-jupyterlab -f Dockerfile.jypyterlab .
docker run -it --rm -p 8888:8888 -v $(pwd):/app mlops-cleancode-jupyterlab
```

The test can be run using the following docker container/commands:

```
docker build -t mlops-cleancode-tests -f Dockerfile.test .
docker run -it --rm mlops-cleancode-tests
```