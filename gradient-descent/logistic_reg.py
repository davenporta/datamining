#!/usr/bin/env python3

"""Implement logistic regression"""

import math
import numpy as np

from symdiff import *


def pad_ones_column(X):
    """Prepend a column of 1s to matrix X"""
    assert len(X.shape) == 2
    return np.hstack((np.array([np.ones(X.shape[0])]).T, X))


def logit(X):
    """AKA the sigmoid function"""
    return 1 / (1 + math.e ** -X)


def y_pred(w_vars, x):
    """Essentially logit(w.T.dot(x))

     We can't simply use dot as we must deal with a list of variables instead
      of an array of values for w.
    """
    return logit(sum(w_vars[i] * x[i] for i in range(len(w_vars))))


def gen_cost(X, y):
    """Create cost function in terms of weights"""
    rows, cols = X.shape
    w_vars = [Var('w{}'.format(i)) for i in range(cols)]

    return Function(
        - 1 / rows * sum(
            y[i] * ln(y_pred(w_vars, X[i])) + (1 - y[i]) * ln(1 - y_pred(w_vars, X[i]))
            for i in range(rows))
        , *w_vars)


class LogisticRegression:
    """Implement logistic regression with gradient descent using the `symdiff` module

    Features:
    * Can handle multi-dimensional data

    Caveats:
    * Classes must be binary.

    Instance Variables:
    * w: The vector [b, w_1, w_2, ..., w_n]
    * errors: A list of the cost function value over time/epochs
    """

    def __init__(self, learning_rate, iterations):
        """Initialize model

        Parameters:
        * learning_rate: Too low, and convergence will take forever. Too high,
                          and you will overshoot.
        * iterations: How many epochs to iterate over
        """
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X_org, y):
        X = pad_ones_column(X_org)
        rows, cols = X.shape

        self.w = np.zeros(cols)
        cost = gen_cost(X, y)
        grad = gradient(cost)

        self.errors = [cost(*self.w)]

        for _ in range(self.iterations):
            grad_value = np.array([evaluate(comp, *self.w) for comp in grad])
            self.w += self.learning_rate * -grad_value
            self.errors.append(cost(*self.w))

    def predict(self, X_org):
        X = pad_ones_column(X_org)
        return np.array([1 if logit(self.w.T.dot(X[i])) > 0.5 else 0
                         for i in range(X.shape[0])])
