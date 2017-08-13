# -*- coding: utf-8 -*-
'''
Головной решатель на базе нейросетки, созданной средствами Lasagne,
для бенчмарка эффективности разных word representation в задаче
определения допустимости N-граммы.

В качестве прототипа использован исходный текст https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py

(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import numpy as np
import time
import theano
import theano.tensor as T
from lasagne.layers import EmbeddingLayer, InputLayer
from lasagne.layers import DenseLayer
import lasagne
import sklearn
import pickle
import colorama


from DatasetVectorizers import WordIndeces_Vectorizer
from DatasetSplitter import split_dataset


batch_size = 1000
embed_dim = 32
num_epochs = 50

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

colorama.init()

dataset_generator = WordIndeces_Vectorizer()
X_data,y_data = dataset_generator.vectorize_dataset()

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


# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.binary_crossentropy(prediction, target_var).mean()
# We could add some weight decay as well here, see lasagne.regularization.
# ...

# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
#updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
updates = lasagne.updates.adam(loss, params, learning_rate=0.01)

inputs = [input_var, target_var]
train_fn = theano.function(inputs=inputs, outputs=[loss], updates=updates)

# Create a loss expression for validation/testing. The crucial difference
# here is that we do a deterministic forward pass through the network,
# disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True)


test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.round_half_away_from_zero(test_prediction), target_var), dtype=theano.config.floatX)
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

predict_fn = theano.function(inputs=[input_var], outputs=test_prediction)

# -----------------------------------------------------------------------

print("Training...")

WEIGHTS_FILE = 'wr_lasagne.model'
best_val_acc = 0.0
no_improvement_counter = 0 # подсчет эпох без улучшения точности на валидации, для early stopping

# We iterate over epochs:
for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0.0
    train_batches = 0
    start_time = time.time()
    for (inputs, targets) in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
        e = train_fn(inputs, targets)
        train_err += e[0]
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0.0
    val_acc = 0.0
    val_batches = 0
    for (inputs, targets) in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    y_pred = predict_fn(X_val)[:,0]
    y_pred = (y_pred > 0.5).astype(int)
    val_acc = sklearn.metrics.accuracy_score(y_val, y_pred)

    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc * 100.0))

    if val_acc > best_val_acc:
        print(colorama.Fore.GREEN + 'New best val_acc={:.2f}, store the model...'.format(val_acc) + colorama.Fore.RESET)
        best_val_acc = val_acc
        no_improvement_counter = 0
        values = lasagne.layers.get_all_param_values(network)
        with open(WEIGHTS_FILE, 'w') as f:
            pickle.dump(values, f)
    else:
        no_improvement_counter += 1
        if no_improvement_counter >= 10:
            print('No improvement during last {} epochs, stop the training'.format(no_improvement_counter))
            break

# --------------------------------------------------------

# Загрузим лучшую модель
with open(WEIGHTS_FILE,'r') as f:
    weights = pickle.load(f)
    lasagne.layers.set_all_param_values(network, weights)


#holdout_err, holdout_acc = val_fn(X_holdout, y_holdout)
#print('holdout_loss={} holdout_acc={}'.format(holdout_err, holdout_acc) )

#y_pred = predict_fn(X_holdout)[:,0]
#y_pred  = (y_pred > 0.5).astype(int)
#acc = sklearn.metrics.accuracy_score( y_holdout, y_pred )
#print('test_loss={} test_acc={}%'.format(test_loss, acc*100.0))

n_hit = 0
n_total = 0
val_batches = 0
for (inputs, targets) in iterate_minibatches(X_holdout, y_holdout, batch_size, shuffle=False):
    y_pred = predict_fn(inputs)[:,0]
    y_pred  = (y_pred > 0.5).astype(int)
    n_hit += sum( y_pred == targets )
    n_total += len(y_pred)

print('Final holdout accuracy={:.2f} %'.format( n_hit*100.0/n_total ))


