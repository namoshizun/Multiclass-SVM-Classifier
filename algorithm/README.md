## Multiclass SVM Classifier

### Introduction

A multiclass SVM classifier implemented in Python. Uses multiprocessing library to train SVM classifier units in parallel and [Cvxopt](http://cvxopt.org) to solve the quadratic programming problem for each classifier unit.

Supports 10-fold cross validation, auto generation of  the confusion matrix and various accuracy measurements (TP, FP, Precision, Recall etc...)



### Run Locally

The source code in master branch depends on Python3.5. You may switch to "python2-compatible" branch for Python2.7 or above. 

**Install dependencies via**:

> pip install -r requirements.txt

**Usage**:

```
usage: main.py [-h]
               [{linear,poly}] [{one_vs_one,one_vs_rest}] [C] [min_lagmult]
               [cross_validate] [evaluate_features] [{dev,prod}]

SVM Classifier

positional arguments:
  {linear,poly}         The kernel function to use
  {one_vs_one,one_vs_rest}
                        The strategy to implement a multiclass SVM. Choose
                        "one_vs_one" or "one_vs_rest"
  C                     The regularization parameter that trades off margin
                        size and training error
  min_lagmult           The support vector's minimum Lagrange multipliers
                        value
  cross_validate        Whether or not to cross validate SVM
  evaluate_features     Will read the cache of feature evaluation results if
                        set to False
  {dev,prod}            Reads dev data in ../input-dev/ if set to dev mode,
                        otherwise looks for datasets in ../input/

optional arguments:
  -h, --help            show this help message and exit
```

**Example Usage**:

> python main.py

The program will load the training and testing datasets in ../input/, perform feature selection without re-computing the feature information gains, trains SVM using C= 1.0, polynomial kernel and One-vs-One strategy. It then saves the predictions of testing dataset into ../output/:

### 