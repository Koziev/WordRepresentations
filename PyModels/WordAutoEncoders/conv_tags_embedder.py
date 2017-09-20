# -*- coding: utf-8 -*-
'''
Сверточная нейросеть для проверки возможности вывода морфологичеких тегов из
character surface слова.
'''

from __future__ import (absolute_import, division, print_function, unicode_literals)
import gc
import sklearn
import codecs
import numpy as np
import os
import pandas as pd
import pickle
import keras
from keras.layers import Dense, Dropout, Input, Flatten, concatenate
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
from keras import backend as K


class TagsEmbedder(object):


    def __init__(self):
        self.BOUNDARY_CHAR = u'\b'
        pass

    @staticmethod
    def normalize_word(word):
        return word.lower()

    def generate_rows(self, grammar_dict, batch_size, mode):
        required_remainder = 0
        if mode==1:
            required_remainder = 1
        elif mode==2:
            required_remainder = None
        batch_index = 0
        batch_count = 0

        X_batch = np.zeros((batch_size, self.max_word_len, self.nb_chars), dtype=np.float32)
        tags_output_size = grammar_dict.get_tags_size()
        y_tags_batch = np.zeros((batch_size, tags_output_size), dtype=bool)

        while True:
            for iword,word in enumerate(grammar_dict.vocab):
                if required_remainder is None or (iword%2)==required_remainder:

                    for tag in grammar_dict.get_tags(word):
                        y_tags_batch[batch_index,tag] = True

                    batch_index += 1

                    if batch_index == batch_size:
                        batch_count += 1
                        #print('mode={} batch_count={}'.format(mode, batch_count))
                        if mode==2:
                            yield X_batch
                        else:
                            yield ({'input_layer': X_batch}, { 'tags': y_tags_batch })

                        # очищаем матрицы порции для новой порции
                        X_batch.fill(0)
                        y_tags_batch.fill(0)
                        batch_index = 0


    def encode_word(self, word):
        return self.BOUNDARY_CHAR+word.replace( u' - ', u'-').replace(u' ', u'_')+self.BOUNDARY_CHAR

    def fit(self, vector_size, grammar_dict, data_folder):
        words_list = list([self.normalize_word(word) for word in grammar_dict.vocab])
        self.nb_words = len(words_list)
        self.max_word_len = max( len(word) for word in words_list )+2 # два граничных символа
        all_chars = set(self.BOUNDARY_CHAR)
        for word in words_list:
            all_chars.update(word)

        self.nb_chars = len(all_chars)
        self.char2index = dict( (c,i) for (i,c) in enumerate(all_chars) )

        input = Input(shape=(self.max_word_len, self.nb_chars), dtype='float32', name='input_layer')

        nb_filters = 64 #self.max_word_len*self.nb_chars

        conv_list = []
        merged_size = 0

        for kernel_size in range(2, 6):
            conv_layer = Conv1D(filters=nb_filters,
                                kernel_size=kernel_size,
                                padding='valid',
                                activation='relu',
                                strides=1)(input)
            conv_layer = GlobalMaxPooling1D()(conv_layer)
            conv_list.append(conv_layer)
            merged_size += nb_filters

        merged = keras.layers.concatenate(inputs=conv_list)
        encoder = Dense(units=int(merged_size/2), activation='relu')(merged)
        encoder = Dense(units=int(merged_size/4), activation='relu')(encoder)
        encoder = Dense(units=vector_size, activation='sigmoid')(encoder)


        decoder_tags = Dense(units=grammar_dict.get_tags_size(), activation='relu' )(encoder)
        decoder_tags = Dense(units=grammar_dict.get_tags_size(), activation='relu', name='tags')(decoder_tags)

        model = Model(inputs=input, outputs=decoder_tags)
        model.compile(loss='mse', optimizer='nadam' )  #, metrics=['accuracy'])

        with open(os.path.join(data_folder,'tags_embedder.arch'),'w') as f:
            f.write(model.to_json())

        model_checkpoint = ModelCheckpoint( os.path.join(data_folder,'tags_embedder.model'),
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='auto')

        early_stopping = EarlyStopping( monitor='val_loss',
                                        patience=10,
                                        verbose=1,
                                        mode='auto')

        batch_size = 100
        nbatch = int(self.nb_words / (2*batch_size))
        print('nbatch={}'.format(nbatch))

        hist = model.fit_generator(generator=self.generate_rows(grammar_dict, batch_size, mode=0),
                            steps_per_epoch=nbatch,
                            epochs=100,
                            verbose=1,
                            callbacks=[model_checkpoint, early_stopping],
                            validation_data=self.generate_rows(grammar_dict, batch_size, mode=1),
                            validation_steps=nbatch,
                            )

        val_loss = hist.history['val_loss']
        epoch_list = list(range(1,len(val_loss)+1))
        df_loss = pd.DataFrame(index=epoch_list, columns=['epoch', 'val_loss'])
        df_loss['epoch'] = epoch_list
        df_loss['val_loss'] = val_loss
        df_loss.to_csv(os.path.join(data_folder,'tags_embedder_losses.csv'), index=False)




        model.load_weights( os.path.join(data_folder,'tags_embedder.model') )

        # теперь нам нужен только энкодер, чтобы получить векторы слов на выходе lstm слоя
        print('Build model2')
        model2 = Model(inputs=input, outputs=encoder)
        model2.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

        print('Predicting with model2')
        y2 = model2.predict_generator( generator=self.generate_rows(grammar_dict, batch_size, mode=2),
                                       steps=int(self.nb_words/batch_size) )

        print('Storing {} matrix with word vectors'.format(y2.shape))
        #np.save( os.path.join(data_folder,'ae_vectors.npy'), y2)
        #with open(os.path.join(data_folder,'ae_words.pkl'), 'w') as f:
        #    pickle.dump(words_list, f)

        with codecs.open(os.path.join(data_folder,'tags_embedder_vectors.dat'), 'w', 'utf-8') as wrt:
            wrt.write('{} {}\n'.format(self.nb_words, vector_size))
            for i,word in enumerate(words_list[:y2.shape[0]]): # не все слова попадут в y2 из-за округления на батчах, но делать батчи=1 слишком тяжко для расчета
                wrt.write(u'{} {}\n'.format(self.encode_word(word), unicode.join( u' ', [str(y) for y in y2[i]])) )

        return


word_vector_size = 64

grdict = GrammarDictionary()
grdict.load(need_w2v=False)

embedder = TagsEmbedder()
embedder.fit( word_vector_size, grdict, '../../data')
