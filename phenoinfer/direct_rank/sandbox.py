from __future__ import print_function

import sys
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

# sys.path.append('..')
import tensorflow as tf
from direct_rank.DirectRanker import directRanker
from direct_rank.helpers import readData, nDCGScorer_cls, MAP_cls

type=["union","go","uberon","mp","intersection"]
for tp in type:
    print(tp)

    x_train, y_train, q_train = readData(data_path="../data/direct_rank/"+tp+"_train.txt", binary=False, at=10, number_features=200, bin_cutoff=1.5, cut_zeros=True)
    # For debugging
    # x_train, y_train, q_train = readData(debug_data=True, binary=True, at=10, number_features=136, bin_cutoff=1.5,
    #                                      cut_zeros=True)

    x_test, y_test, q_test = readData(data_path="../data/direct_rank/"+tp+"_test.txt", binary=False, at=10, number_features=200, bin_cutoff=1.5, cut_zeros=True)
    # For debugging
    # x_test, y_test, q_test = readData(debug_data=True, binary=True, at=10, number_features=136, bin_cutoff=1.5,
    #                                   cut_zeros=True)


    def lambda_cost(nn, y0):
        return tf.reduce_mean(tf.log(1+tf.exp((1+nn)/2))-(1+nn)/2)


    # Load directRanker, train, and test
    dr = directRanker(
        feature_activation=tf.nn.tanh,
        ranking_activation=tf.nn.tanh,
        # max_steps=10000,
        # For debugging
        cost=lambda_cost,
        max_steps=5000,
        print_step=500,
        start_batch_size=3,
        end_batch_size=5,
        start_qids=20,
        end_qids=100,
        feature_bias=True,
        hidden_layers=[100, 50]
    )

    dr.fit(x_train, y_train, ranking=True)

    nDCGScorer_cls(dr, x_test, y_test, at=100)
    MAP_cls(dr, x_test, y_test)
