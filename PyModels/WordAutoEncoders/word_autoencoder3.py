# -*- coding: utf-8 -*-
'''
Генерация представления слов через тренировку расширенного автоэнкодера для
использования в https://github.com/Koziev/WordRepresentations.

Сетка обучается как автоэнкодер на словах с дополнительными target'ами, чтобы
скрытый слой содержал информацию о: 1) битовом векторе морфологических признаках
2) w2v векторах.


Результатом работы является словарь соответствий слов и скрытых
представлений.

Тренировочный датасет генирируется батчами, так как целиком в память он для
сложных target'ов не поместится.

(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import (absolute_import, division, print_function, unicode_literals)
import gc
import sklearn
import codecs
import numpy as np
import os
import pickle
from keras.utils import plot_model
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


class AutoEncoder(object):
    def __init__(self):
        pass

    def normalize_word(self, word):
        return word.lower()

    def generate_rows(self, grammar_dict, batch_size, mode):
        required_remainder = 0
        if mode==1:
            required_remainder = 1
        elif mode==2:
            required_remainder = None
        batch_index = 0
        batch_count = 0

        tags_output_size = grammar_dict.get_tags_size()
        w2v_output_size = grammar_dict.get_w2v_size()

        X_batch = np.zeros((batch_size, self.max_word_len, self.nb_chars), dtype=np.float32)
        y_word_batch = np.zeros((batch_size, self.max_word_len, self.nb_chars), dtype=np.float32)
        y_tags_batch = np.zeros((batch_size, tags_output_size), dtype=bool)
        y_w2v_batch = np.zeros((batch_size, w2v_output_size), dtype=np.float32)

        while True:
            for iword,word in enumerate(grammar_dict.vocab):
                if required_remainder is None or (iword%2)==required_remainder:
                    for ichar,c in enumerate(word):
                        X_batch[batch_index, ichar, self.char2index[c]] = True
                        y_word_batch[batch_index, ichar, self.char2index[c]] = True

                    for tag in grammar_dict.get_tags(word):
                        y_tags_batch[batch_index,tag] = True

                    y_w2v_batch[batch_index, :] = grammar_dict.get_w2v(word)

                    batch_index += 1

                    if batch_index == batch_size:
                        batch_count += 1
                        #print('mode={} batch_count={}'.format(mode, batch_count))
                        if mode==2:
                            yield X_batch
                        else:
                            yield ({'input_layer': X_batch}, { 'word': y_word_batch, 'tags': y_tags_batch, 'w2v': y_w2v_batch })

                        # очищаем матрицы порции для новой порции
                        X_batch.fill(0)
                        y_word_batch.fill(0)
                        y_tags_batch.fill(0)
                        y_w2v_batch.fill(0)
                        batch_index = 0


    def encode_word(self, word):
        return word.replace( u' - ', u'-').replace(u' ', u'_')

    def fit(self, vector_size, grammar_dict, data_folder):
        words_list = list([self.normalize_word(word) for word in grammar_dict.vocab])
        self.nb_words = len(words_list)
        self.max_word_len = max( len(word) for word in words_list )
        all_chars = set()
        for word in words_list:
            all_chars.update(word)

        self.nb_chars = len(all_chars)
        self.char2index = dict( (c,i) for (i,c) in enumerate(all_chars) )

        input = Input(shape=(self.max_word_len, self.nb_chars), dtype='float32', name='input_layer')
        encoder = LSTM( vector_size, input_shape=(self.max_word_len, self.nb_chars), return_sequences=False )(input)

        decoder_word = RepeatVector(self.max_word_len)(encoder)
        decoder_word = LSTM(vector_size, return_sequences=True)(decoder_word)
        decoder_word = TimeDistributed(Dense(self.nb_chars))(decoder_word)
        decoder_word = Activation('softmax', name='word')(decoder_word)

        decoder_tags = Dense(units=grammar_dict.get_tags_size(), name='tags')(encoder)

        decoder_w2v = Dense(units=grammar_dict.get_w2v_size(), activation='tanh', name='w2v')(encoder)


        model = Model(inputs=input, outputs=[decoder_word, decoder_tags, decoder_w2v])
        model.compile(loss={'word':'categorical_crossentropy', 'tags':'mse', 'w2v':'mse'},
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        plot_model(model, to_file=os.path.join(data_folder,'ae_model.png'))

        with open(os.path.join(data_folder,'word_autoencoder.arch'),'w') as f:
            f.write(model.to_json())

        model_checkpoint = ModelCheckpoint( os.path.join(data_folder,'word_autoencoder.model'),
                                            monitor='val_word_acc',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='auto')

        early_stopping = EarlyStopping( monitor='val_word_acc',
                                        patience=10,
                                        verbose=1,
                                        mode='auto')

        batch_size = 100
        nbatch = int(self.nb_words / (2*batch_size))
        print('nbatch={}'.format(nbatch))

        hist = model.fit_generator(generator=self.generate_rows(grammar_dict, batch_size, mode=0),
                            steps_per_epoch=nbatch,
                            epochs=20,
                            verbose=2,
                            callbacks=[model_checkpoint, early_stopping],
                            validation_data=self.generate_rows(grammar_dict, batch_size, mode=1),
                            validation_steps=nbatch,
                            )
        with open(os.path.join(data_folder,'ae_history.pkl'), 'w') as f:
            pickle.dump(hist,f)

        model.load_weights( os.path.join(data_folder,'word_autoencoder.model') )

        # теперь нам нужен только энкодер, чтобы получить векторы слов на выходе lstm слоя
        print('Build model2')
        model2 = Model(inputs=input, outputs=encoder)
        model2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        print('Predicting with model2')
        y2 = model2.predict_generator( generator=self.generate_rows(grammar_dict, batch_size, mode=2),
                                       steps=int(self.nb_words/batch_size) )

        print('Storing {} matrix with word vectors'.format(y2.shape))
        #np.save( os.path.join(data_folder,'ae_vectors.npy'), y2)
        #with open(os.path.join(data_folder,'ae_words.pkl'), 'w') as f:
        #    pickle.dump(words_list, f)

        with codecs.open(os.path.join(data_folder,'ae_vectors.dat'), 'w', 'utf-8') as wrt:
            wrt.write('{} {}\n'.format(self.nb_words, vector_size))
            for i,word in enumerate(words_list[:y2.shape[0]]): # не все слова попадут в y2 из-за округления на батчах, но делать батчи=1 слишком тяжко для расчета
                wrt.write(u'{} {}\n'.format(self.encode_word(word), unicode.join( u' ', [str(y) for y in y2[i]])) )

        return




word_vector_size = 32

grdict = GrammarDictionary()
grdict.load(need_w2v=True)

autoencoder = AutoEncoder()
autoencoder.fit( word_vector_size, grdict, '../../data')
