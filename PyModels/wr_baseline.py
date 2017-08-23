# -*- coding: utf-8 -*-
'''
Baseline solver в задаче определения допустимости N-граммы.
Используется простое запоминание N-грамм в тренировочном наборе.
(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import gc
import sklearn.model_selection
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from DatasetVectorizers import BaseVectorizer
from DatasetSplitter import split_dataset


REPRESENTATIONS = 'word_indeces'


dataset_generator = BaseVectorizer.get_dataset_generator(REPRESENTATIONS)
X_data,y_data = dataset_generator.vectorize_dataset()
X_train,  y_train, X_val, y_val, X_holdout, y_holdout = split_dataset(X_data, y_data )
gc.collect()

print('X_train.shape={} X_val.shape={} X_holdout.shape={}'.format(X_train.shape, X_val.shape, X_holdout.shape))


class BaselineEstimator(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        print('Store {}-grams from training dataset'.format(X.shape[1]))
        self.known_ngrams = set()
        for i in range(X.shape[0]):
            if y[i]==1:
                ngram = tuple(X_train[i, :])
                self.known_ngrams.add(ngram)
        print('{} unique {}-grams stored'.format(len(self.known_ngrams), X.shape[1]))

    def predict(self, X):
        y_pred = []
        for i in range(X_holdout.shape[0]):
            ngram = tuple(X_holdout[i,:])
            y = 1 if ngram in self.known_ngrams else 0
            y_pred.append(y)

        return y_pred

# -------------------------------------------------

model = BaselineEstimator()
model.fit(X_train, y_train)

print('Predicting...')
y_pred = model.predict(X_holdout)
test_loss = sklearn.metrics.log_loss( y_holdout, y_pred )
acc = sklearn.metrics.accuracy_score( y_holdout, y_pred )

print('test_loss={} test_acc={}'.format(test_loss, acc))

