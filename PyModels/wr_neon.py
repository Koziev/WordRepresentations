# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Train a small multi-layer perceptron with fully connected layers on MNIST data.
This example has some command line arguments that enable different neon features.
Examples:
    python examples/mnist_mlp.py -b gpu -e 10
        Run the example for 10 epochs using the NervanaGPU backend
    python examples/mnist_mlp.py --eval_freq 1
        After each training epoch, process the validation/test data
        set through the model and display the cost.
    python examples/mnist_mlp.py --serialize 1 -s checkpoint.pkl
        After every iteration of training, dump the model to a pickle
        file named "checkpoint.pkl".  Changing the serialize parameter
        changes the frequency at which the model is saved.
    python examples/mnist_mlp.py --model_file checkpoint.pkl
        Before starting to train the model, set the model state to
        the values stored in the checkpoint file named checkpoint.pkl.
"""

from neon.callbacks.callbacks import Callbacks
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine
from neon.layers import LookupTable
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, CrossEntropyBinary, Misclassification
from neon.util.argparser import NeonArgparser
from neon.data import ArrayIterator
from neon import logger as neon_logger

from DatasetVectorizers import BaseVectorizer
from DatasetSplitter import split_dataset
import CorpusReaders


REPRESENTATIONS = 'w2v' # 'word_indeces' | 'w2v' | 'w2v_tags'

corpus_reader = CorpusReaders.ZippedCorpusReader('../data/corpus.txt.zip')
#corpus_reader = CorpusReaders.TxtCorpusReader(r'f:\Corpus\Raw\ru\tokenized_w2v.txt')


dataset_generator = BaseVectorizer.get_dataset_generator(REPRESENTATIONS)
X_data,y_data = dataset_generator.vectorize_dataset(corpus_reader=corpus_reader)
X_train,  y_train, X_val, y_val, X_holdout, y_holdout = split_dataset(X_data, y_data )

train_set = ArrayIterator(X=X_train, y=y_train)
valid_set = ArrayIterator(X=X_val, y=y_val)

# setup weight initialization function
init_norm = Gaussian(loc=0.0, scale=0.01)

if REPRESENTATIONS=='word_indeces':
    input_word_dims = 32
    nb_words = dataset_generator.nb_words
    ngram_arity = X_train.shape[1]

    dense_size = input_word_dims*ngram_arity
    layers = [ neon.layers.LookupTable( vocab_size=nb_words, embedding_dim=input_word_dims  ),
              Affine(nout=dense_size, init=init_norm, activation=Rectlin()),
              BatchNorm(),
              Affine(nout=dense_size/2, init=init_norm, activation=Rectlin()),
              BatchNorm(),
              Affine(nout=1, init=init_norm, activation=Logistic(shortcut=True))]
    
else:    
    input_size = X_train.shape[1]
    layers = [Affine(nout=input_size, init=init_norm, activation=Rectlin()),
              BatchNorm(),
              Affine(nout=input_size/2, init=init_norm, activation=Rectlin()),
              BatchNorm(),
              Affine(nout=input_size/3, init=init_norm, activation=Rectlin()),
              BatchNorm(),
              Affine(nout=1, init=init_norm, activation=Logistic(shortcut=True))]

# setup cost function as CrossEntropy
cost = GeneralizedCost(costfunc=CrossEntropyBinary())

# setup optimizer
optimizer = GradientDescentMomentum(
    0.1, momentum_coef=0.9, stochastic_round=args.rounding)

# initialize model object
mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, eval_set=valid_set ) #, **args.callback_args)

# run fit
mlp.fit(train_set,
        optimizer=optimizer,
        num_epochs=args.epochs,
        cost=cost,
        callbacks=callbacks)
error_rate = mlp.eval(valid_set, metric=Misclassification())
neon_logger.display('Misclassification error = %.1f%%' % (error_rate * 100))
