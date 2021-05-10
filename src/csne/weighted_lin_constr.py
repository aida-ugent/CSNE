#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 02/05/2020

from __future__ import division
import numpy as np
from scipy.sparse import *
from abc import abstractmethod


class WeightLinConstr:
    """
    Abstract class which defines the methods that all priors need to implement.
    In this case we assume we have almost O(n^2) space.
    """

    @abstractmethod
    def sc_mult(self, val):
        """ Multiply F with scalar val element-wise and return the result. """

    @abstractmethod
    def mat_sum_mult(self, mat):
        """ Returns the sum of an element-wise multiplication between F and mat. """

    @abstractmethod
    def sqmat_sum_mult(self, mat):
        """ Returns the sum of an element-wise multiplication between F^2 and mat. """

    @abstractmethod
    def get_row(self, i):
        """Returns the f values of row i"""

    @abstractmethod
    def get_elem(self, src, dst):
        """Returns the f value of an element i,j"""

    @abstractmethod
    def get_mat(self):
        """Returns F as sparse matrix"""


class Uniform(WeightLinConstr):
    def __init__(self, A):
        # If adj matrix is not sparse, make it sparse
        if not issparse(A):
            A = csr_matrix(A)
        self.__n = A.shape[0]
        self.__F = self._compute_f(A)
        mask = csr_matrix(A.astype(bool))
        self.maskdF = np.ones(mask.nnz) * self.__F
        self.maskdF2 = np.power(self.maskdF, 2)

    @staticmethod
    def _compute_f(A):
        # Compute matrix B as the 'mask'
        B = A.copy()
        B.data = (B.data + 1) / 2.0
        B.eliminate_zeros()
        C = A.copy()
        C.data = np.abs(C.data)
        return B.sum() / C.sum()

    def sc_mult(self, val):
        F = np.ones((self.__n, self.__n)) * (self.__F * val)
        return F

    def mat_sum_mult(self, mat):
        return mat.sum() * self.__F

    def sqmat_sum_mult(self, mat):
        return mat.sum() * self.__F ** 2

    def get_row(self, i):
        return np.ones(self.__n) * self.__F

    def get_elem(self, src, dst):
        return self.__F

    def get_mat(self):
        raise NotImplementedError()


class CommonNeigh(WeightLinConstr):
    def __init__(self, A):
        # If adj matrix is not sparse, make it sparse
        if not issparse(A):
            A = csr_matrix(A)
        self.__F = self._compute_f(A)
        mask = csr_matrix(A.astype(bool))
        r, c = mask.nonzero()
        self.maskdF = mask.multiply(self.__F)[r, c].A.ravel()
        self.maskdF2 = np.power(self.maskdF, 2)

    @staticmethod
    def _compute_f(A):
        F = A.dot(
            A.T
        )  # for dir networks A.dot(A) give in degree and A.dot(A.T) gives out degree
        F.setdiag(0)
        F.eliminate_zeros()
        F.sort_indices()
        return F.multiply(1 / F.max())

    def sc_mult(self, val):
        """ Multiply F with scalar val element-wise and return the result. """
        return self.__F.multiply(val).A

    def mat_sum_mult(self, mat):
        """ Returns the sum of an element-wise multiplication between F and mat. """
        return (self.__F.multiply(mat)).sum()

    def sqmat_sum_mult(self, mat):
        """ Returns the sum of an element-wise multiplication between F^2 and mat. """
        aux = self.__F.multiply(self.__F)
        return (aux.multiply(mat)).sum()

    def get_row(self, i):
        return self.__F[i, :].A.ravel()

    def get_elem(self, src, dst):
        return self.__F[src, dst]

    def get_mat(self):
        return self.__F


class cpp(CommonNeigh):
    @staticmethod
    def _compute_f(A):
        """ Counts the number of ++ wedges in A. """
        # Compute matrix B as the 'mask'
        B = A.copy()
        B.data = (B.data + 1) / 2.0
        B.eliminate_zeros()
        # Compute pp wedges
        F = B.dot(B.T)
        F.setdiag(0)
        F.eliminate_zeros()
        return F


class cmm(CommonNeigh):
    @staticmethod
    def _compute_f(A):
        """ Counts the number of -- wedges in A. """
        # Compute matrix B as the 'mask'
        B = A.copy()
        B.data = np.abs((B.data - 1) / 2.0)
        B.eliminate_zeros()
        # Compute mm wedges
        F = B.dot(B.T)
        F.setdiag(0)
        F.eliminate_zeros()
        return F


class cpm(CommonNeigh):
    @staticmethod
    def _compute_f(A):
        """ Counts the number of +- wedges in A. """
        # Compute matrix B as the 'mask'
        B = A.copy()
        B.data = np.abs(B.data)
        # Compute cn and subtract those which have the same sign, pp or mm. this leaves you with wedges pm.
        cn = B.dot(B.T)
        F = (cn - A.dot(A.T)) / 2.0
        F.setdiag(0)
        F.eliminate_zeros()
        return F
