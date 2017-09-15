# -*- coding: utf-8 -*-
"""
Отрисовка схемы вычислительного графа для нейросетки-автоэнкодера.
Предполагается, что word_autoencoder3.py отработал и сохранил
архитектуру нейросети в файл word_autoencoder.arch.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.layers.core import Activation
from keras.layers.recurrent import LSTM
from keras.layers.core import RepeatVector, Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from keras.models import model_from_json
from keras.utils import plot_model

model = model_from_json( open(os.path.join('../../data', 'word_autoencoder.arch'), 'r').read() )
plot_model(model, to_file=open(os.path.join('../../data', 'model.png') ) )
