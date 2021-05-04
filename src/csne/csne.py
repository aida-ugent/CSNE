#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian and Yoosof Mashayekhi
# Contact: alexandru.mara@ugent.be, yoosof.mashayekhi@ugent.be
# Date: 02/05/2020
#
# The code is base on the original implementation of:
# Conditional Network Embeddings (CNE) (Copyright (c) Ghent University)
# by Bo Kang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
import time
from collections import defaultdict
from os.path import join

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange
import time


class CSNE:
    def __init__(self, A, params, prior, samplemask=None):
        self.__A = A
        self.__n = A.shape[0]
        self.__d = params['d']
        self.__k = params['k']
        self.__s1 = params['s1']
        self.__s2 = params['s2']
        self.__params = params
        self.__diff_inv_sp2 = (1/self.__s1**2 - 1/self.__s2**2)
        self.__s1_div_s2 = self.__s1/self.__s2
        self.__prior = prior
        self._compute_adj_list(A)
        self._sample_ids(samplemask)

    def _compute_adj_list(self, A):
        __csr_A = A.tocsr()
        self.__adj_list = []
        for i in range(__csr_A.shape[0]):
            self.__adj_list.append(
                __csr_A.indices[__csr_A.indptr[i]:__csr_A.indptr[i+1]])

    def get_sample_ids(self, i):
        return self.__sample_ids_list[i]

    def get_edge_masks(self, i):
        return self.__edge_masks[i]

    def _sample_ids(self, samplemask=None):
        self.__sample_ids_list = []
        self.__edge_masks = []
        self.__num_pos_sample_list = []
        self.__prior_ratio = []

        for i in trange(self.__n, desc='Neighborhood sampling'):
            if samplemask is None:
                samples = []
                pos_ids = self.__adj_list[i]
                num_pos_samples = min(len(pos_ids), self.__k)
                if num_pos_samples != 0:
                    samples.extend(pos_ids[np.random.choice(len(pos_ids), num_pos_samples)])
                self.__num_pos_sample_list.append(num_pos_samples)

                samples.extend(list(set(
                    np.random.randint(self.__n,
                                      size=(1, 2 * self.__k - num_pos_samples))[0]) - set(pos_ids) - set({i})))

                edge_mask = np.zeros_like(samples).astype(bool)
                edge_mask[:num_pos_samples] = True

                self.__sample_ids_list.append(samples)
                self.__edge_masks.append(edge_mask)
                P = self.__prior.get_row_probability(i, samples)
                self.__prior_ratio.append((1 - P) / P)
            else:
                row = np.squeeze(np.array(self.__A[i, :].todense()))
                mask = np.squeeze(np.array(samplemask[i, :].todense()))
                mask[i] = False
                sample_ids = np.where(mask)[0]

                self.__sample_ids_list.append(sample_ids)
                aux = row[sample_ids]
                self.__edge_masks.append(aux > 0.5)
                P = self.__prior.get_row_probability(i, sample_ids)
                self.__prior_ratio.append((1-P)/P)

    def _compute_squared_dist(self, X, target_id, sample_ids):
        return np.sum((X[target_id, :] - X[sample_ids, :])**2, axis=1).T

    def _eval_obj(self, X):
        obj = 0.
        for i in range(self.__n):
            sample_ids = self.__sample_ids_list[i]
            edge_mask = self.__edge_masks[i]

            diff = (X[i, :] - X[sample_ids, :]).T
            D = np.sum(diff**2, axis=0)
            P_aij_X = 1. / (1 + self.__s1_div_s2 * self.__prior_ratio[i] * np.exp(self.__diff_inv_sp2 * D / 2))

            obj += np.sum(np.log(P_aij_X[edge_mask] + 1e-20)) + np.sum(np.log(1-P_aij_X[~edge_mask] + 1e-20))

        return obj

    def _eval_grad(self, X, epsilon=0.01):
        grad = np.zeros_like(X)
        for i in range(self.__n):
            sample_ids = self.__sample_ids_list[i]
            edge_mask = self.__edge_masks[i]

            diff = (X[i, :] - X[sample_ids, :]).T
            D = np.sum(diff**2, axis=0)
            P_aij_X = 1. / (1 + self.__s1_div_s2 * self.__prior_ratio[i] * np.exp(self.__diff_inv_sp2 * D / 2))

            grad_i = 2 * self.__diff_inv_sp2 * ((P_aij_X - edge_mask) * diff).T
            grad[i, :] += np.sum(grad_i, axis=0)
            grad[sample_ids, :] -= grad_i
        return grad

    def compute_row_posterior(self, row_id, col_ids, X=None):
        if X is None:
            X = self.__emb
        P = self.__prior.get_row_probability(row_id, col_ids)
        D = self._compute_squared_dist(X, row_id, col_ids)
        return 1. / (1 + self.__s1_div_s2 * (1 - P) / P * np.exp(self.__diff_inv_sp2 * D / 2))

    def optimizer_adam(self, X, num_epochs=100, alpha=0.001, beta_1=0.9,
                       beta_2=0.9999, eps=1e-8, verbose=True, epsilon=0.01):
        m_prev = np.zeros_like(X)
        v_prev = np.zeros_like(X)
        for epoch in trange(num_epochs, desc='Training CSNE'):
            grad = -self._eval_grad(X, epsilon=epsilon)

            # Adam optimizer
            m = beta_1*m_prev + (1-beta_1)*grad
            v = beta_2*v_prev + (1-beta_2)*grad**2

            m_prev = m.copy()
            v_prev = v.copy()

            m = m/(1-beta_1**(epoch+1))
            v = v/(1-beta_2**(epoch+1))
            X -= alpha*m/(v**.5 + eps)

            if verbose:
                if epoch % 1 == 0:
                    grad_norm = np.sum(grad**2)**.5
                    tqdm.write('Epoch: {:d}, gradient norm: {:.4f}'.format(epoch, grad_norm))
                if epoch == num_epochs-1:
                    grad_norm = np.sum(grad**2)**.5
                    obj = self._eval_obj(X)
                    tqdm.write('Epoch: {:d}, obj: {:.4f}, gradient norm: {:.4f}'.format(epoch, -obj, grad_norm))
        return X

    def fit(self, verbose=True, X0=None):
            if X0 is None:
                X = np.random.randn(self.__n, self.__d)
            else:
                X = X0
            optimizer = self.__params["optimizer"]
            if optimizer["name"] == 'adam':
                self.__emb = self.optimizer_adam(X, num_epochs=optimizer['max_iter'], alpha=optimizer['lr'],
                                                 verbose=verbose, epsilon=self.__params.get("epsilon", 0))
            else:
                raise ValueError('optimizer {:s} is not implemented.'.format(optimizer["name"]))

    def predict(self, E):
        edge_dict = defaultdict(list)
        ids_dict = defaultdict(list)
        for i, edge in enumerate(E):
            edge_dict[edge[0]].append(edge[1])
            ids_dict[edge[0]].append(i)

        pred = []
        ids = []
        for u in edge_dict.keys():
            pred.extend(self.compute_row_posterior(u, edge_dict[u]))
            ids.extend(ids_dict[u])

        return [p for _,p in sorted(zip(ids, pred))]

    def get_embedding(self):
        return self.__emb

    def get_adj_row(self, row_id):
        return self.__A[row_id, :]

    def get_posterior_row(self, row_id):
        return self.compute_row_posterior(row_id, range(self.__n))

    def get_parameters(self):
        params = self.__params.copy()
        params['sample_ids_list'] = self.__sample_ids_list
        params['edge_masks'] = self.__edge_masks
        return params
