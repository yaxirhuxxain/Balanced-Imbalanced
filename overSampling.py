# -*- coding: utf-8 -*-

# Author Yasir Hussain (yaxirhuxxain@yahoo.com)

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import tensorflow as tf
from tensorflow import keras

from util import get_metrics

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(sess)


class Trainer(object):

    def __init__(self, context_size=20,
                 n_dim=300,
                 n_epochs=2 ** 32 - 1,
                 n_batch=32,
                 n_drop=0.25,
                 learner="RNN",
                 optimizer="adam",
                 learn_rate=0.001,
                 activation="softmax",
                 loss="sparse_categorical_crossentropy",
                 patience=10,
                 out_dir=None):

        self.context_size = context_size
        self.n_dim = n_dim
        self.n_epochs = n_epochs  # Yes, 2**32 is technically infinity
        self.init_epoch = 0  # initialay is 1
        self.n_batch = n_batch
        self.n_drop = n_drop
        self.vocab_size = None
        self.learn_rate = learn_rate
        self.learner = learner
        self.optimizer = optimizer
        self.activation = activation
        self.loss = loss
        self.patience = patience
        self.word_idx = None

        if out_dir:
            self.out_dir = out_dir
        else:
            root_path = os.getcwd()
            self.out_dir = os.path.join(root_path, os.path.join('saveModels', self.learner))

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Embedding(self.vocab_size, self.n_dim, input_length=self.context_size))

        if self.learner == 'RNN':
            model.add(keras.layers.SimpleRNN(self.n_dim))

        elif self.learner == 'LSTM':
            model.add(keras.layers.LSTM(self.n_dim))

        elif self.learner == 'GRU':
            model.add(keras.layers.GRU(self.n_dim))

        elif self.learner == 'CNN':
            model.add(keras.layers.Conv1D(filters=self.n_dim, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.GlobalMaxPooling1D())

        else:
            sys.exit("Invalid Learner")

        if self.n_drop:
            model.add(keras.layers.Dropout(self.n_drop))

        if self.activation == 'softmax':
            model.add(keras.layers.Dense(self.vocab_size, activation='softmax'))
        elif self.activation == 'sigmoid':
            model.add(keras.layers.Dense(self.vocab_size, activation='sigmoid'))
        else:
            print("Unknown activation: %r", self.learner)
            sys.exit(2)

        model.summary()

        return model

    def compile_model(self, model):

        if self.optimizer == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(lr=self.learn_rate)
        elif self.optimizer == 'adam':
            optimizer = keras.optimizers.Adam(lr=self.learn_rate)
        else:
            print("Unknown optimizer: %r", self.optimizer)
            sys.exit(2)

        metric_list = get_metrics()
        model.compile(loss=self.loss, optimizer=optimizer, metrics=metric_list)

        print(" Model Successfully Compiled")

    def sampler(self, sampler, X_train, y_train):

        # oversampling
        if sampler == 'PSMOTE':
            import smote_variants as sv
            print(f"Before Resampling(X_train): {len(X_train)}")
            oversampler = sv.MulticlassOversampling(sv.polynom_fit_SMOTE(random_state=7))
            X_train_res, y_train_res = oversampler.sample(X_train, y_train)
            print(f"After Resampling(X_train_res) with {sampler}: {len(X_train_res)}")

        elif sampler == 'ProWSyn':
            import smote_variants as sv
            print(f"Before Resampling(X_train): {len(X_train)}")
            oversampler = sv.MulticlassOversampling(sv.ProWSyn(n_neighbors=1, L=1, n_jobs=-1, random_state=7))
            X_train_res, y_train_res = oversampler.sample(X_train, y_train)
            print(f"After Resampling(X_train_res) with {sampler}: {len(X_train_res)}")

        return X_train_res, y_train_res

    def save_model_summary(self, model):
        from contextlib import redirect_stdout
        with open(os.path.join(self.out_dir, 'model-summary.txt'), 'w') as f:
            with redirect_stdout(f):
                model.summary()

    def build_voacb(self, tokens_stream, remove_singleton=True):
        word_count = {}
        for word in tokens_stream:
            try:
                word_count[word] += 1
            except:
                word_count[word] = 1

        word_count = dict(sorted(word_count.items(), key=lambda kv: kv[1], reverse=True))

        word_idx = {}
        word_idx["unk"] = 0
        if remove_singleton:
            count = 0
            for key, value in word_count.items():
                if value <= 1:
                    continue
                else:
                    count += 1
                    word_idx[key] = count
        else:
            for count, word in enumerate(word_count.keys()):
                word_idx[word] = count + 1

        vocab_size = len(word_idx)
        return vocab_size, word_idx, word_count

    def build_sequences(self, files_list):
        sequences = []
        for file in files_list:
            tokens_stream = file.split()
            for i in range(0, len(tokens_stream)):
                sequence = tokens_stream[i - self.context_size:i + 1]
                if len(sequence) > self.context_size:
                    encoded = [self.word_idx[word] if word in self.word_idx else 0 for word in sequence]
                    sequences.append(encoded)
        sequences = list(filter(None, sequences))
        return sequences