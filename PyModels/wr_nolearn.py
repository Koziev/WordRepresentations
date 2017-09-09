# -*- coding: utf-8 -*-
'''
Головной решатель на базе нейросетки, созданной средствами nolearn/Lasagne,
для бенчмарка эффективности разных word representation в задаче определения
допустимости N-граммы.
(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet
import numpy as np
import sklearn.model_selection
import time

import theano
import theano.tensor as T
from lasagne.layers import EmbeddingLayer, InputLayer
from lasagne.layers import DenseLayer
import lasagne
import nolearn
from nolearn.lasagne import NeuralNet
import pickle

from DatasetVectorizers import WordIndeces_Vectorizer
from DatasetSplitter import split_dataset
import CorpusReaders


batch_size = 256
embed_dim = 32
num_epochs = 100

# ----------------------------------------------------------------------------



def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    n = len(inputs)
    if shuffle:
        indices = np.arange(n)
        np.random.shuffle(indices)
    for start_idx in range(0, n - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# ----------------------------------------------------------------------------

corpus_reader = CorpusReaders.ZippedCorpusReader('../data/corpus.txt.zip')
#corpus_reader = CorpusReaders.TxtCorpusReader(r'f:\Corpus\Raw\ru\tokenized_w2v.txt')


dataset_generator = WordIndeces_Vectorizer()
X_data,y_data = dataset_generator.vectorize_dataset(corpus_reader=corpus_reader)

X_train,  y_train, X_val, y_val, X_holdout, y_holdout = split_dataset(X_data, y_data )

print('X_train.shape={} X_val.shape={} X_holdout.shape={}'.format(X_train.shape, X_val.shape, X_holdout.shape))


nb_words = dataset_generator.nb_words
ngram_arity = X_train.shape[1]
input_size = ngram_arity

n_dense = input_size * embed_dim

input_var = T.imatrix('inputs')
target_var = T.ivector('targets')

a1 = InputLayer(shape=(batch_size,input_size), input_var=input_var)
net = EmbeddingLayer(a1, input_size=nb_words, output_size=embed_dim)
net = DenseLayer(net, n_dense, nonlinearity=lasagne.nonlinearities.rectify)
net = DenseLayer(net, n_dense/2, nonlinearity=lasagne.nonlinearities.rectify)
#net = DenseLayer(net, n_dense/4, nonlinearity=lasagne.nonlinearities.rectify)
net = DenseLayer(net, 1, nonlinearity=lasagne.nonlinearities.sigmoid)
network = net

# -------------------------------------------------------------

WEIGHTS_FILE = 'wr_nolearn.model'

# Этот простой callback даст возможность следить за прогрессом в обучении
# в терминале.
best_val_acc = 0.0
no_improvement_count = 0
def on_epoch_finished( net, history ):
    global best_val_acc, no_improvement_count
    train_loss = history[-1]['train_loss']
    valid_loss = history[-1]['valid_loss']
    duration = history[-1]['dur']

    # если качество улучшилось, то надо бы сохранять веса модели в файл
    y_pred = net.predict_proba(X_val)
    y_pred = y_pred[:, 0]
    y_pred = (y_pred > 0.5).astype(int)
    acc = sklearn.metrics.accuracy_score(y_val, y_pred)
    if acc>best_val_acc:
        best_val_acc = acc
        no_improvement_count = 0
        with open(WEIGHTS_FILE,'w') as f:
            pickle.dump(net,f)
    else:
        no_improvement_count += 1
        if no_improvement_count>10:
            raise StopIteration()

    print('Epoch #{} finished: eta={} train_loss={} valid_loss={} valid_acc={}'.format(len(history), duration, train_loss, valid_loss, acc ))


net = NeuralNet( net,
                 update_learning_rate=0.01,
                 objective_loss_function=lasagne.objectives.binary_crossentropy,
                 train_split=nolearn.lasagne.base.TrainSplit(eval_size=0.1),
                 on_epoch_finished=[ on_epoch_finished ]
                )

net.fit( X_train, y_train )

# загружаем лучший вариант весов
with open(WEIGHTS_FILE,'r') as f:
    net = pickle.load(f)

y_pred = net.predict_proba( X_holdout )[:,0]
y_pred  = (y_pred > 0.5).astype(int)
acc = sklearn.metrics.accuracy_score( y_holdout, y_pred )

print('test_acc={}'.format(acc))
