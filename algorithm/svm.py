from util import linear
from cvxopt import matrix as cvmat
from cvxopt.solvers import qp, options
from util import kernel_function
import numpy as np
import pickle

options['show_progress'] = False  # less verbose


class HyperplanSeparator:
    def __init__(self, X, Y, K, alphas, min_lagmult, kernel):
        """
        Representation of the hyperplan separator/predicator
        :param alphas: all Lagrange multipliers
        :param min_lagmult: support vector's minimum Lagrange multiplier value
        :param kernel: kernel function
        """
        sv_selector = alphas > min_lagmult
        sv_alphas = alphas[sv_selector]
        sv_Y = Y[sv_selector]
        self.sv_X = X[sv_selector]
        self.kernel = kernel

        # compute bias and weight according to : # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
        # partial weight is a computation cache which can be used to compute weights
        self.partial_weight = sv_alphas * sv_Y
        self.bias = np.mean(sv_Y - np.sum(self.partial_weight * K[sv_selector][:, sv_selector], axis=1))

    def predict(self, X):
        if X[0] is not np.ndarray:
            X = [X]
        results = [np.sum(self.partial_weight * self.kernel(x, self.sv_X.T)) for x in X]
        return self.bias + results

class SVM:
    def __init__(self, kernel, C, min_lagmult, strategy):
        self.kernel = kernel_function(kernel)
        self.C = C
        self.min_lagmult = min_lagmult
        self.strategy = strategy
        self.hyper_separator = None

    @property
    def __config(self):
        return {
            'min_lagmult': self.min_lagmult,
            'kernel': self.kernel
        }

    def qp_components(self, X, Y):
        """
        Refenrece: http://cvxopt.org/userguide/coneprog.html#quadratic-cone-programs

        Build quadratic programming components that can be directly used by cvxopt.solver.qp
        cvxopt requires the qp problem defined in the following form:
            min (1/2)*x^T*P*x + q^T*x
            subject to Gx <= h
                       Ax  = b

        The formulation of the qp problem to be solved SVM with soft-margin is therefore defined as:
            argmin_a (1/2)\sum_i,j(y_i*y_j*a_i*a_j*K(x_i*x_j)) - \sum_i(a_i)
            s.t, 0 <= a_i <= C, for i = 1~N
            and  \sum_i(a_iy_i) = 0

        :param X: training data as np.matrix
        :param Y: training labels as np.matrix
        :return:
        """
        kernel = self.kernel
        n_samples, n_features = X.shape
        n_ones = np.ones(n_samples)

        # optimisation problem terms
        K = kernel(X, X.T)
        P = cvmat(np.outer(Y, Y) * K)
        q = cvmat(n_ones * -1.)
        A = cvmat(Y.reshape(1, -1))
        b = cvmat(0.0)

        # constraint terms
        lower_bound = np.zeros(n_samples)
        upper_bound = n_ones * self.C
        G = cvmat(np.vstack((np.diag(n_ones * -1), np.diag(n_ones))))
        h = cvmat(np.hstack((lower_bound, upper_bound)))

        return P, q, G, h, A, b, K

    def fit(self, X, Y):
        P, q, G, h, A, b, K = self.qp_components(X, Y)
        # solve the QP problem to get Lagrange multipliers
        result = qp(P, q, G, h, A, b)
        alphas = np.ravel(result['x'])

        # build the SVM separator that uses only the support vectors
        # whose lagrange multiplier value is greater than the treshold
        self.hyper_separator = HyperplanSeparator(X, Y, K, alphas, **self.__config)
        return self

    def predict(self, X):
        """
        One-vs-One: return the actual 1 or -1 class
        One-vs-Rest: return the value of decision function directly
        """
        if self.strategy == 'one_vs_one':
            return np.sign(self.hyper_separator.predict(X))
        if self.strategy == 'one_vs_rest':
            return self.hyper_separator.predict(X)
