"""
@User: sandruskyi

"""
from __future__ import division

import logging
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import label_binarize
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import pinv


class MultinomialRegression(BaseEstimator, RegressorMixin):
    def __init__(self, weights_0=None, method='Full', initializer='identity',
                 reg_format=None, reg_lambda=0.0, reg_mu=None, reg_norm=False,
                 ref_row=True, optimizer='auto'):
        if method not in ['Full', 'Diag', 'FixDiag']:
            raise ValueError(f"Unknown method {method}")

        self.weights_0 = weights_0
        self.method = method
        self.initializer = initializer
        self.reg_format = reg_format
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu
        self.reg_norm = reg_norm
        self.ref_row = ref_row
        self.optimizer = optimizer

    def __setup(self):
        self.classes = None
        self.weights_ = self.weights_0
        self.weights_0_ = self.weights_0

    @property
    def coef_(self):
        return self.weights_[:, :-1]

    @property
    def intercept_(self):
        return self.weights_[:, -1]

    def predict_proba(self, X):
        X_ = np.hstack([X, np.ones((X.shape[0], 1))])
        return self._softmax(np.dot(X_, self.weights_.T))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X, y, *args, **kwargs):
        self.__setup()
        n_samples = X.shape[0]
        X_ = np.hstack([X, np.ones((n_samples, 1))])
        self.classes = np.unique(y)
        k = len(self.classes)

        # Label binarization
        target = label_binarize(y, classes=self.classes)
        if k == 2:
            target = np.hstack([1 - target, target])

        # Regularization scaling
        if self.reg_norm:
            if self.reg_mu is None:
                self.reg_lambda = self.reg_lambda / (k * (k + 1))
            else:
                self.reg_lambda = self.reg_lambda / (k * (k - 1))
                self.reg_mu = self.reg_mu / k

        # Initial weights
        self.weights_0_ = self._get_initial_weights(k)
        X_flat = X_
        target_flat = target

        # Choose optimizer
        if self.optimizer == 'newton' or (self.optimizer == 'auto' and k <= 36):
            self.weights_ = self._newton_update(self.weights_0_, X_flat, target_flat, k)
        else:  # L-BFGS-B
            res = fmin_l_bfgs_b(
                func=self._objective,
                x0=self.weights_0_,
                fprime=self._gradient,
                args=(X_flat, target_flat, k),
                maxiter=1024
            )
            self.weights_ = self._reshape_weights(res[0], k)

        return self

    def _get_initial_weights(self, k):
        if self.weights_0_ is not None:
            return self.weights_0_
        if self.initializer == 'identity':
            if self.method in ['Full', None]:
                raw = np.hstack([np.eye(k), np.zeros((k, 1))])
            elif self.method == 'Diag':
                raw = np.hstack([np.ones(k).reshape(-1, 1), np.zeros((k, 1))])
            elif self.method == 'FixDiag':
                raw = np.ones((k, 1))
            else:
                raise ValueError
        else:
            raw = np.zeros((k, k + 1))
        return raw.ravel()

    def _reshape_weights(self, params, k):
        if self.method in ['Full', None]:
            raw_weights = params.reshape(-1, k + 1)
        elif self.method == 'Diag':
            raw_weights = np.hstack([np.diag(params[:k]), params[k:].reshape(-1, 1)])
        elif self.method == 'FixDiag':
            raw_weights = np.hstack([np.eye(k) * params[0], np.zeros((k, 1))])
        else:
            raise ValueError(f"Unknown method {self.method}")

        if self.ref_row:
            raw_weights = raw_weights - raw_weights[-1, :]
        return raw_weights

    def _softmax(self, X):
        shiftx = X - np.max(X, axis=1, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _objective(self, params, X, y, k):
        weights = self._reshape_weights(params, k)
        outputs = self._softmax(np.dot(X, weights.T))
        loss = -np.mean(np.sum(y * np.log(outputs + 1e-12), axis=1))
        if self.reg_lambda > 0:
            loss += self.reg_lambda * np.sum(weights[:, :-1] ** 2)
        if self.reg_mu is not None:
            loss += self.reg_mu * np.sum(weights[:, -1] ** 2)
        return loss

    def _gradient(self, params, X, y, k):
        weights = self._reshape_weights(params, k)
        outputs = self._softmax(np.dot(X, weights.T))
        grad = np.dot((outputs - y).T, X)
        return grad.ravel()

    def _newton_update(self, weights_0, X, y, k, maxiter=50, tol=1e-8):
        weights = weights_0.copy()
        for i in range(maxiter):
            outputs = self._softmax(np.dot(X, weights.reshape(-1, k + 1).T))
            grad = np.dot((outputs - y).T, X).ravel()
            # Approx Hessian as identity for simplicity
            hess_inv = np.eye(len(grad))
            update = hess_inv.dot(grad)
            weights -= update
            if np.linalg.norm(update) < tol:
                break
        return weights.reshape(-1, k + 1)