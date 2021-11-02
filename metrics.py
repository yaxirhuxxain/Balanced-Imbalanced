# -*- coding: utf-8 -*-

# Author Yasir Hussain (yaxirhuxxain@yahoo.com)

import sys
import warnings

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


def top_k_acc(y_true, y_pred, k=1):
    return tf.reduce_mean(tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=k))


def top_k(y_pred, k=5):
    _, ranks = tf.nn.top_k(y_pred, k=k)
    return ranks.numpy()  # works on eager execution


def Top_k_Accuracy(y_true, y_pred):
    top_1 = top_k_acc(y_true, y_pred, k=1)
    top_2 = top_k_acc(y_true, y_pred, k=2)
    top_3 = top_k_acc(y_true, y_pred, k=3)
    top_5 = top_k_acc(y_true, y_pred, k=5)
    top_10 = top_k_acc(y_true, y_pred, k=10)
    return [top_1.numpy(), top_2.numpy(), top_3.numpy(), top_5.numpy(), top_10.numpy()]  # works on egger exicution


def Precision_Recall_Fscore_Sklearn(y_true, y_pred, flag="weighted"):
    y_cls = top_k(y_pred, k=1)
    if flag == 'weighted':
        result = precision_recall_fscore_support(y_true, y_cls, average='weighted')
    elif flag == 'macro':
        result = precision_recall_fscore_support(y_true, y_cls, average='macro')
    elif flag == 'micro':
        result = precision_recall_fscore_support(y_true, y_cls, average='micro')
    else:
        sys.exit('Invlaid flag. Choose one of these (weighted,macro,micro)')
    precision = result[0]
    recall = result[1]
    f1 = result[2]

    return [precision, recall, f1]


def MRR_Score(y_true, y_pred, k=5):
    y_true = tf.Variable(y_true)
    y_true = tf.expand_dims(y_true, 1)
    _, ranks = tf.nn.top_k(y_pred, k=k)

    # ranks = tf.cast(ranks, tf.int64)
    ranks = tf.cast(ranks, y_true.dtype)

    mrr = tf.where(tf.equal(ranks, y_true))[:, 1]  # works but removes the indexes which are not found
    mrr = 1 / (mrr + 1)

    # fixing removed indexes
    unk_ranks = tf.zeros([tf.shape(y_true)[0] - tf.shape(mrr)[0], ],
                         'float64')  # make a new tensor with zeros to fill the not found ranks
    total_ranks = tf.concat([mrr, unk_ranks],
                            0)  # adjust the not found ranks to corrrectly calculate mean reciprocal ranks

    return tf.reduce_mean(total_ranks).numpy()
