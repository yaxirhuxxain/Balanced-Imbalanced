# -*- coding: utf-8 -*-

# Author Yasir Hussain (yaxirhuxxain@yahoo.com)

import csv
import os
import sys

import tensorflow.keras.backend as K


def top_1_acc(y_true, y_pred, k=1):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)


def top_2_acc(y_true, y_pred, k=2):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)


def top_3_acc(y_true, y_pred, k=3):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)


def top_5_acc(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)


def top_10_acc(y_true, y_pred, k=10):
    return K.mean(K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k), axis=-1)


def categorical_top_1_acc(y_true, y_pred, k=1):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)


def categorical_top_2_acc(y_true, y_pred, k=2):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)


def categorical_top_3_acc(y_true, y_pred, k=3):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)


def categorical_top_5_acc(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)


def categorical_top_10_acc(y_true, y_pred, k=10):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)


def get_metrics_categorical():
    metrics_list = ['accuracy',
                    categorical_top_1_acc,
                    categorical_top_2_acc,
                    categorical_top_3_acc,
                    categorical_top_5_acc,
                    categorical_top_10_acc]
    return metrics_list


def get_metrics():
    metrics_list = ['accuracy',
                    top_1_acc,
                    top_2_acc,
                    top_3_acc,
                    top_5_acc,
                    top_10_acc]
    return metrics_list


def sec_to_hms(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    periods = [('hours', hours), ('minutes', minutes), ('seconds', seconds)]
    time_string = ', '.join('{} {}'.format(value, name)
                            for name, value in periods
                            if value)
    return time_string


def score_to_csv(model_folder, scores, type):
    _file = os.path.join(model_folder, type + '_scores.csv')

    if os.path.isfile(_file):
        with open(_file, 'a+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(scores)
    else:
        with open(_file, 'a+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if type == 'Top_K':
                filewriter.writerow(['Learner', 'Project', 'Top-1', 'Top-2', 'Top-3', 'Top-5', 'Top-10'])
                filewriter.writerow(scores)

            if type == 'PRF':
                filewriter.writerow(['Learner', 'Project', 'Precision', 'Recall', 'F1'])
                filewriter.writerow(scores)

            if type == 'MRR':
                filewriter.writerow(['Learner', 'Project', 'MRR'])
                filewriter.writerow(scores)


class Logging:
    def __init__(self, filename):
        self.out_file = open(filename, "w")
        self.old_stdout = sys.stdout
        # this object will take over `stdout`'s job
        sys.stdout = self

    # executed when the user does a `print`
    def write(self, text):
        self.old_stdout.write(text)
        self.out_file.write(text)

    def flush(self):
        pass

    # executed when `with` block begins
    def __enter__(self):
        return self

    # executed when `with` block ends
    def __exit__(self, type, value, traceback):
        # we don't want to log anymore. Restore the original stdout object.
        sys.stdout = self.old_stdout
