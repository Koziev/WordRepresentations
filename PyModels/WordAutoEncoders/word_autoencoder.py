# -*- coding: utf-8 -*-
'''
Генерация представления слов через тренировку vanilla автоэнкодера для
использования в https://github.com/Koziev/WordRepresentations.
Результатом работы является словарь соответствий слов и скрытых
представлений.

ATT: тренировочный датасет генерируется весь в памяти, поэтому может потребоваться 32 Гб
для выполнения.

(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import (absolute_import, division, print_function, unicode_literals)
import gc
import sklearn
import numpy as np
import codecs
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
from grammar_dictionary import GrammarDictionary
import os
import pickle


class AutoEncoder(object):
    def __init__(self):
        pass

    def normalize_word(self, word):
        return word.lower()

    def fit(self, vector_size, grammar_dict, data_folder):
        words_list = list([self.normalize_word(word) for word in grammar_dict.vocab])
        nb_words = len(words_list)
        max_word_len = max( len(word) for word in words_list )
        all_chars = set()
        for word in words_list:
            all_chars.update(word)

        nb_chars = len(all_chars)
        char2index = dict( (c,i) for (i,c) in enumerate(all_chars) )

        X_train = np.zeros((nb_words, max_word_len, nb_chars), dtype=np.float32)
        y_train = np.zeros((nb_words, max_word_len, nb_chars), dtype=np.float32)

        for iword,word in enumerate(words_list):
            for ichar,c in enumerate(word):
                X_train[iword, ichar, char2index[c]] = True
                y_train[iword, ichar, char2index[c]] = True

        input = Input(shape=(max_word_len, nb_chars), dtype='float32', name='input_layer')
        encoder = LSTM( vector_size, input_shape=(max_word_len, nb_chars), return_sequences=False )(input)
        decoder = RepeatVector(max_word_len)(encoder)
        decoder = LSTM(vector_size, return_sequences=True)(decoder)
        decoder = TimeDistributed(Dense(nb_chars))(decoder)
        decoder = Activation('softmax')(decoder)

        model = Model(inputs=input, outputs=decoder)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        with open(os.path.join(data_folder,'word_autoencoder.arch'),'w') as f:
            f.write(model.to_json())

        model_checkpoint = ModelCheckpoint( os.path.join(data_folder,'word_autoencoder.model'),
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='auto')

        early_stopping = EarlyStopping( monitor='val_loss',
                                        patience=10,
                                        verbose=1,
                                        mode='auto')

        batch_size = 150
        model.fit(X_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=100,
                  validation_split=0.1,
                  verbose=2,
                  callbacks=[model_checkpoint, early_stopping] )

        model.load_weights( os.path.join(data_folder,'word_autoencoder.model') )

        # теперь нам нужен только энкодер, чтобы получить векторы слов на выходе lstm слоя
        print('Build model2')
        model2 = Model(inputs=input, outputs=encoder)
        model2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        print('Predicting with model2')
        y2 = model2.predict( X_train )

        print('Storing {} matrix with word vectors'.format(y2.shape))
        with codecs.open(os.path.join(data_folder,'ae_vectors.dat'), 'w', 'utf-8') as wrt:
            wrt.write('{} {}\n'.format(self.nb_words, vector_size))
            for i,word in enumerate(words_list[:y2.shape[0]]):
                wrt.write(u'{} {}\n'.format(self.encode_word(word), unicode.join( u' ', [str(y) for y in y2[i]])) )

        return




word_vector_size = 32

grdict = GrammarDictionary()
grdict.load()

autoencoder = AutoEncoder()
autoencoder.fit( word_vector_size, grdict, '../../data')
