# -*- coding: utf-8 -*-
'''
Решение задачи https://github.com/Koziev/WordRepresentations с помощью нейросетки на Keras.

Проверяется только один вариант представления слов - sparse distributed representation.

Так как проверяемые представления слов (sparse distributed representations) имеют
очень большую размерность (1024 или больше), то сгенерировать сразу тренировочную
матрицу для всех сэмплов в памяти не возможно. Поэтому генерируем датасеты порциями
и используем специальные методы в Keras модели - fit_generator и evaluate_generator.
Генератор порций реализован в функции generate_rows. За основу берется представление
слов индексами в лексиконе (класс WordIndeces_Vectorizer). Далее каждая порция создается
простой заменой индекса на вектор соответствующего слова.

(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import gc
import sklearn
import codecs
import random
import sys
import numpy as np
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from DatasetVectorizers import WordIndeces_Vectorizer
from DatasetSplitter import split_dataset
import CorpusReaders



# арность N-грамм
NGRAM_ORDER = 3

# кол-во сэмплов в датасете
NB_SAMPLES = 10000000

# Архитектура нейросети
NET_ARCH = 'MLP' # 'MLP' | 'CNN'

# -----------------------------------------------------------------------

def generate_rows(X, sdr_vec_len, y, index2word, word2sdr, batch_size):
    """
    Генератор порций данных для обучения и валидации.
    
    
    :param X: полный списк N-грамм, каждое слово представлено целочисленным индексом
    :param sdr_vec_len: длина вектора представления слова
    :param y: список эталонных значений классификации для N-грамм в X
    :param index2word: словарь для получения текстового представления слова по его индексу в X
    :param word2sdr: словарь для получения вектора слова
    :param batch_size: размер генерируемых порций
    :return: пара матриц X_batch и y_batch
    """
    nrow = X.shape[0]
    ngram_arity = X.shape[1]
    input_size = ngram_arity*sdr_vec_len
    nb_batch = int(nrow/batch_size)

    X_batch = np.zeros( (batch_size, input_size), dtype=np.bool )
    y_batch = np.zeros( (batch_size), dtype=np.bool )

    #try:
    while True:
        indeces = range(nrow)
        random.shuffle(indeces)

        for ibatch in range(nb_batch):

            X_batch.fill(0)
            y_batch.fill(0)

            for irow in range(batch_size):
                ii = ibatch*batch_size + irow
                ngram = X[ii,:]
                for j in range(ngram_arity):
                    word = index2word[ngram[j]]
                    if word in word2sdr:
                        X_batch[irow, j*sdr_vec_len:(j+1)*sdr_vec_len] = word2sdr[word]

                y_batch[irow] = y[ii]

            yield ({'input_layer': X_batch}, {'output_layer': y_batch})

    #except:
    #    print( 'Unexpected error: {}'.format( sys.exc_info()[0] ) )
    #    raise

# -----------------------------------------------------------------------

def build_model( input_size ):

    x_input = Input(shape=(input_size,), dtype='float32', name='input_layer')
    ndense = input_size

    print('Building MLP...')
    net = Dense(units=ndense, activation='relu')(x_input)
    #net = BatchNormalization()(net)
    net = Dense(units=int(ndense / 2), activation='relu')(net)
    #net = BatchNormalization()(net)
    net = Dense(units=int(ndense / 3), activation='relu')(net)
    #net = BatchNormalization()(net)
    net = Dense( units=int(ndense/4), activation='relu' )(net)
    #net = BatchNormalization()(net)
    net = Dense(units=1, activation='sigmoid', name='output_layer')(net)

    model = Model(inputs=x_input, outputs=net)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


# -----------------------------------------------------------------------

corpus_reader = CorpusReaders.ZippedCorpusReader('../data/corpus.txt.zip')
#corpus_reader = CorpusReaders.TxtCorpusReader(r'f:\Corpus\Raw\ru\tokenized_w2v.txt')


dataset_generator = WordIndeces_Vectorizer()
X_data,y_data = dataset_generator.vectorize_dataset(corpus_reader=corpus_reader, ngram_order=NGRAM_ORDER, nb_samples=NB_SAMPLES)
gc.collect()
X_train,  y_train, X_val, y_val, X_holdout, y_holdout = split_dataset(X_data, y_data )
ngram_arity = dataset_generator.get_ngram_arity()
gc.collect()

word2id = dataset_generator.get_vocabulary()
# Нам нужны SDR векторы только для этих слов.
ngram_words = set(word2id.keys())
index2word = dict( (i,w) for (w,i) in word2id.iteritems() )


# Грузим SDR.
# TODO: вынести загрузчик в отдельный класс.
sdr_path = r'/home/eek/polygon/WordSDR2/sdr.dat'
print('Loading SDRs...')
word2sdr = dict()
with codecs.open(sdr_path, 'r', 'utf-8') as rdr:
    line0 = rdr.readline().strip()
    toks = line0.split(u' ')
    nword = int(toks[0])
    veclen = int(toks[1])
    for line in rdr:
        tx = line.strip().split()
        word = tx[0]
        if word in ngram_words:
            vec = [(True if float(z) > 0.0 else False) for z in tx[1:]]
            vec = np.asarray(vec, dtype=np.bool)
            word2sdr[word] = vec

input_size = veclen * ngram_arity

# Создаем сетку нужной архитектуры
model = build_model(input_size)

weights_filename = 'wr_keras.model'

print('Train...')

# Генерируем батчи из обучающего набора.
# Перед каждой эпохой тасуем обучающие N-граммы.

batch_size = 200
steps_per_epoch = int(X_train.shape[0]/batch_size)


model_checkpoint = ModelCheckpoint( weights_filename, monitor='val_acc', verbose=1,
                                   save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='auto')


model.fit_generator( generator=generate_rows(X_train, veclen, y_train, index2word, word2sdr, batch_size),
                     steps_per_epoch=steps_per_epoch,
                     epochs=100,
                     verbose=1,
                     callbacks=[model_checkpoint, early_stopping],
                     validation_data=generate_rows(X_val, veclen, y_val, index2word, word2sdr, batch_size),
                     validation_steps=int(X_val.shape[0]/batch_size),
                     )
                     #class_weight=None,
                     #max_queue_size=10,
                     #workers=1,
                     #use_multiprocessing=False,
                     #initial_epoch=0 )


model.load_weights(weights_filename)

print('Final evaluation...')
res = model.evaluate_generator( generator=generate_rows(X_holdout, veclen, y_holdout, index2word, word2sdr, batch_size),
                                steps=int(X_holdout.shape[0]/batch_size) )

print('holdout acc={}'.format(res[model.metrics_names.index('acc')]))
