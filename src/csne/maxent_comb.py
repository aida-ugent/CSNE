#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 02/05/2020

from __future__ import division
from __future__ import print_function
import numpy as np
from csne.weighted_lin_constr import *
from tqdm import trange, tqdm
from collections import defaultdict
import time


class MaxentCombined:

    def __init__(self, A, mask, func, memory='quadratic', reg_val=0.9):
        """
        Initializes the maxent combined model

        Parameters
        ----------
        A : Scipy sparse matrix
            A sparse adjacency matrix representing a certain network.
        func : list
            A list containing one or a combination of {'cpp', 'cmm', 'cpm'}.
        memory : string, optional
            A string indicating if the method should use 'linear' or 'quadratic' memory. Quadratic actually only uses
            O(E) memory, same as Adj matrix. Default is 'quadratic'.
        reg_val : float, optional
            A float indicating how much to reduce the confidence about pos edges being +1 and neg edges being -1.
            Default is 0.9.
        """
        A.data[A.data == 1] = reg_val
        A.data[A.data == -1] = -reg_val
        self.__n = A.shape[0]
        self.__2n = 2 * self.__n
        self.__nfuncs = len(func)
        self.__memory = memory
        self.__mask = mask
        self.__F = self._select_f(A, func)          # array with as many priors as __nfuncs
        self.__x = None
        self.__cs = self._get_constraints(A)

    def _select_f(self, A, func):
        print("Initializing MaxentCombined with the following priors: {}".format(func))
        fs = list()
        for fi in range(self.__nfuncs):
            fs.append(self._select(A, func[fi]))
        return fs

    @staticmethod
    def _select(A, func):
        if func == 'cpp':
            return cpp(A)

        elif func == 'cpm':
            return cpm(A)

        elif func == 'cmm':
            return cmm(A)

        else:
            raise ValueError("Prior not implemented.")

    def predict(self, E):
        """
        Returns the predictions for the given set of (src, dst) pairs.

        Parameters
        ----------
        E : iterable
            An iterable of (src, dst) pairs.

        Returns
        -------
        scores : list
            The probabilities in [0,1] of having -1 or +1 links between the (src, dst) pairs, in the same order as E.

        Raises
        ------
        AttributeError
            If the method has not been fitted.
        """
        edge_dict = defaultdict(list)
        ids_dict = defaultdict(list)
        for i, edge in enumerate(E):
            edge_dict[edge[0]].append(edge[1])
            ids_dict[edge[0]].append(i)

        pred = []
        ids = []
        for u in edge_dict.keys():
            pred.extend(self.get_row_probability(u, edge_dict[u]))
            ids.extend(ids_dict[u])

        return [p for _, p in sorted(zip(ids, pred))]

    def get_posterior(self):
        """
        Returns the posterior probability matrix.

        Returns
        -------
        P : ndarray
            The posterior probability matrix.

        Raises
        ------
        AttributeError
            If the method has not been fitted.
        """
        if self.__x is not None:
            return self._get_posterior()
        else:
            raise AttributeError("Maxent Combined has not been fitted. Use <class>.fit()")

    def fit(self, optimizer='newton', lr=1.0, max_iter=100, tol=0.0001, verbose=False):
        """
        Fits the Maxent Combined model.

        Parameters
        ----------
        optimizer : basestring, optional
            A string indicating the optimizer to use. For now only `newton` is available. Default is `newton`.
        lr : float, optional
            The learning rate. Default is 1.0.
        max_iter : int, optional
            The maximum number of iteration the optimization will be run for. Default is 100.
        tol : float, optional
            Triggers early stop if gradient norm is below this value. Default is 0.001.
        verbose : bool
            If True information is printed in each iteration of the optimization process.

        Raises
        ------
        ValueError
            If the the memory constraint or optimizer values are not correct.
        """
        # Initial condition
        x = np.zeros(self.__2n + self.__nfuncs)

        # Compute
        if optimizer == 'newton':
            if self.__memory == 'quadratic':
                self.__x = self._optimizer_newton_quadratic_sp(x, lr, max_iter, tol, verbose)
            else:
                raise ValueError('Incorrect memory constraint. Options are `quadratic` and `linear`')
        else:
            raise ValueError('Optimizer {:s} is not implemented.'.format(optimizer))

    def _optimizer_newton_quadratic_sp(self, x, alpha_init=1.0, max_iter=100, tol=0.001, verbose=False):
        """
        Computes teh MaxEnt prior. Uses hessian information o speed up convergence.
        """
        alpha = alpha_init

        # Initialize P
        P = self.__mask.copy()
        aux = P.data

        # Initialize the objective
        obj = (self.__n ** 2 * 0.6931 - self.__n * 0.6931) - np.dot(x, self.__cs)

        # Get row and col indices of mask
        r_idx, c_idx = self.__mask.nonzero()
        c_idx = self.__n + c_idx

        # Iterate
        for i in trange(max_iter, desc='Fitting prior'):

            # Reset alpha every few iterations
            if i % 10 == 0:
                alpha = alpha_init

            # Compute gradient
            P.data = aux / (aux + 1)
            grad = self._get_sums_sp(P) - self.__cs
            P.data = P.data / (aux + 1)
            hess = self._get_sums_sp(P, f_pow=True)
            delta = grad / (hess + 0.00000001)
            imp = np.dot(delta, grad) * 0.0001

            # Greedy search for best alpha
            while True:
                x_test = x - alpha * delta
                v = x_test[r_idx] + x_test[c_idx]
                for fi in range(self.__nfuncs):
                    v += self.__F[fi].maskdF * x_test[self.__2n + fi]
                obj_test = np.sum(np.logaddexp(0, v)) - np.dot(x_test, self.__cs)
                if obj_test <= obj - (alpha * imp):
                    x = x_test
                    aux = np.exp(v)
                    obj = obj_test
                    break
                alpha = alpha / 2.0

            # Show the norms of the gradient and objective
            if verbose:
                #print(np.linalg.norm(grad))
                tqdm.write("[iter {}] Gradient norm: {:.2f}".format(i, np.linalg.norm(grad)))
                tqdm.write("[iter {}] Objective: {:.2f}".format(i, obj))

            if np.linalg.norm(grad, np.inf) < tol:
                break

        tqdm.write("Final gradient norm: {:.2f}".format(np.linalg.norm(grad)))
        tqdm.write("Final objective: {:.2f}".format(obj))
        return x

    def _get_sums_sp(self, P, f_pow=False):
        """ Returns the sums over cols and rows of P as well as sum over all elems of P * F in a single array. """
        aux = np.zeros(self.__2n + self.__nfuncs)
        aux[:self.__n] = P.sum(axis=1).T.A.ravel()
        aux[self.__n:self.__2n] = P.sum(axis=0).A.ravel()
        if f_pow:
            for fi in range(self.__nfuncs):
                aux[self.__2n + fi] = np.dot(self.__F[fi].maskdF2, P.data)
        else:
            for fi in range(self.__nfuncs):
                aux[self.__2n + fi] = np.dot(self.__F[fi].maskdF, P.data)
        return aux

    def _get_posterior(self):
        """ Returns the full posterior probability matrix computed from the lambdas. """
        P = np.array([self.__x[:self.__n]]).T + self.__x[self.__n:self.__2n]
        for fi in range(self.__nfuncs):
            P += self.__F[fi].sc_mult(self.__x[self.__2n + fi])
        P = np.exp(P)
        P = np.divide(P, (1 + P))
        np.fill_diagonal(P, 0)
        return P

    def get_row_probability(self, row_id, col_ids):
        """ Returns the posterior probability for a given row and set of columns. """
        if self.__x is not None:
            p_row = self.__x[row_id] + np.array(self.__x[self.__n + np.array(col_ids)])
            for fi in range(self.__nfuncs):
                p_row += self.__F[fi].get_row(row_id)[col_ids] * self.__x[self.__2n + fi]
            p = np.exp(p_row)
            p = p / (1 + p)
            return p * np.invert(col_ids == row_id)
        else:
            raise AttributeError("Maxent Combined has not been fitted. Use <class>.fit()")

    def _get_elem_posterior(self, src, dst):
        """ Returns the probability of linking a (src, dst) pair. """
        if src == dst:
            return 0
        else:
            p = self.__x[src] + self.__x[self.__n + dst]
            for fi in range(self.__nfuncs):
                p += self.__F[fi].get_elem(src, dst) * self.__x[self.__2n + fi]
            p = np.exp(p)
            return p / (1 + p)

    def _get_constraints(self, A):
        """ Returns the initial constraints for rows, cols and F matrices as computed from the Adj matrix."""
        B = A.copy()
        B.data = (B.data + 1)/2.0
        B.eliminate_zeros()
        aux = np.zeros(self.__2n + self.__nfuncs)
        aux[:self.__n] = B.sum(axis=1).T.A.ravel()          # Row sums
        aux[self.__n:self.__2n] = B.sum(axis=0).A.ravel()   # Col sums
        for fi in range(self.__nfuncs):
            aux[self.__2n + fi] = self.__F[fi].mat_sum_mult(B)  # Full sum
        return aux
