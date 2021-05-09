#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian and Yoosof Mashayekhi
# Contact: alexandru.mara@ugent.be, yoosof.mashayekhi@ugent.be
# Date: 02/05/2020

from __future__ import absolute_import
import pickle
import time
import argparse
import numpy as np
import networkx as nx
from csne.maxent_comb import MaxentCombined
from csne.csne import CSNE
from scipy.sparse import *


def parse_args():
    """ Parses CSNE arguments. """

    parser = argparse.ArgumentParser(description="Run CSNE.")

    # Input/output parameters
    parser.add_argument('--inputgraph', nargs='?',
                        default='../../data/hp-relations.csv',
                        help='Input graph path')
    
    parser.add_argument('--output', nargs='?',
                        default='network.emb',
                        help='Path where the embeddings will be stored.')

    parser.add_argument('--tr_e', nargs='?', default=None,
                        help='Path of the input train edges. Default None (in this case returns embeddings)')

    parser.add_argument('--tr_pred', nargs='?', default='tr_pred.csv',
                        help='Path where the train predictions will be stored. Default tr_pred.csv')

    parser.add_argument('--te_e', nargs='?', default=None,
                        help='Path of the input test edges. Default None.')

    parser.add_argument('--te_pred', nargs='?', default='te_pred.csv',
                        help='Path where the test predictions will be stored. Default te_pred.csv')

    # Prior related parameters
    parser.add_argument('--prior_tricount', type=int, default=1,
                        help='Toogles triangle count use in prior. (1) use triangles and node polarity, '
                             '(0) only node polarity. Default is 1.')

    parser.add_argument('--prior_learning_rate', type=float, default=1.0,
                        help='Learning rate for prior. Default is 1.0.')

    parser.add_argument('--prior_epochs', type=int, default=10,
                        help='Training epochs for prior. Default is 100.')

    parser.add_argument('--prior_tol', type=float, default=0.0001,
                        help='Early stop prior fit if grad norm is below this value. Default is 0.0001.')

    parser.add_argument('--prior_regval', type=float, default=0.9,
                        help='Regularization value, reduces the certainty about 1s and -1s. Default is 0.9')

    # Embedding related parameters
    parser.add_argument('--use_csne', type=int, default=1,
                        help='Toogle CSNE use. (1) use CSNE, (0) use MaxEnt prior only. Default is 1.')

    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for CSNE. Default is 0.1.')

    parser.add_argument('--epochs', type=int, default=500,
                        help='Training epochs for CSNE. Default is 500.')

    parser.add_argument('--s1', type=float, default=1,
                        help='Sigma 1. Default is 1.')

    parser.add_argument('--s2', type=float, default=2,
                        help='Sigma 2. Default is 2.')

    parser.add_argument('--dimension', type=int, default=2,
                        help='Dimensionality of the CSNE embeddings. Default is 2.')

    # Other parameters
    parser.add_argument('--delimiter', default=',',
                        help='Delimiter used in the input files.')

    parser.add_argument('--directed', action='store_true',
                        help='If specified, network treated as directed. Default is undirected.')
    parser.set_defaults(directed=False)

    parser.add_argument('--verbose', action='store_true',
                        help='Determines the verbosity level of the output.')
    parser.set_defaults(verbose=False)

    return parser.parse_args()


def main_helper(args):
    """ Main of CSNE. """

    # Load edgelist
    E = np.loadtxt(args.inputgraph, delimiter=args.delimiter, dtype=int)

    # Create a graph
    G = nx.DiGraph()

    # We make sure the graph edges have corresponding weights +1 or -1
    G.add_weighted_edges_from(E[:, :3])

    # Get adj matrix of the graph
    if args.directed:
        tr_A = nx.adjacency_matrix(G).astype(float)
    else:
        tr_A = nx.adjacency_matrix(G.to_undirected()).astype(float)

    # Compute mask with 1 for all nonzeros in tr_A
    mask = csr_matrix(tr_A.astype(bool))

    # Fit MaxEnt prior
    start = time.time()
    if args.prior_tricount:
        mc = MaxentCombined(tr_A.tocsr(), mask, ['cpp', 'cpm', 'cmm'], 'quadratic', args.prior_regval)
    else:
        mc = MaxentCombined(tr_A.tocsr(), mask, [], 'quadratic', args.prior_regval)
    mc.fit(optimizer='newton', lr=args.prior_learning_rate, max_iter=args.prior_epochs, tol=args.prior_tol,
           verbose=args.verbose)
    predict_obj = mc

    # Fit CSNE
    if args.use_csne:
        tr_A[tr_A == -1] = 0
        tr_A.eliminate_zeros()
        opt_params = {'name': 'adam', 'lr': args.learning_rate, 'max_iter': args.epochs}
        params = {'d': args.dimension, 'k': 100, 's1': args.s1, 's2': args.s2, 'optimizer': opt_params, 'epsilon': 0.0}
        csne = CSNE(tr_A, params, mc, mask)
        csne.fit()
        predict_obj = csne
    print("Computation time: {}".format(time.time() - start))

    # Read the train edges and compute predictions
    start = time.time()
    if args.tr_e is not None:
        train_edges = np.loadtxt(args.tr_e, delimiter=args.delimiter, dtype=int)
        pred_tr = predict_obj.predict(train_edges)
        np.savetxt(args.tr_pred, pred_tr, delimiter=args.delimiter)

        # Read the test edges and run predictions
        if args.te_e is not None:
            test_edges = np.loadtxt(args.te_e, delimiter=args.delimiter, dtype=int)
            pred_te = predict_obj.predict(test_edges)
            np.savetxt(args.te_pred, pred_te, delimiter=args.delimiter)

    # If tr_e!=None or te_e!=None, output predictions for those edges. 
    # Else, store the embeddings (if use_csne=1) or store posterior (if use_csne=0)
    else:
        if args.use_csne:
            print("Saving CSNE node embeddings...")
            np.savetxt(args.output, predict_obj.get_embedding(), delimiter=args.delimiter)
        else:
            print("Saving MaxEnt posterior...")
            np.savetxt(args.output, predict_obj.get_posterior(), delimiter=args.delimiter)
    print('Prediction time: {}'.format(time.time()-start))


def main():
    args = parse_args()
    main_helper(args)


if __name__ == "__main__":
    main()
