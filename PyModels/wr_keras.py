# -*- coding: utf-8 -*-
'''
Головной решатель на базе нейросетки keras для бенчмарка эффективности разных word representation
в задаче определения допустимости N-граммы.
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



REPRESENTATIONS = 'w2v_tags' # 'word_indeces' | 'w2v' | 'w2v_tags'


dataset_generator = BaseVectorizer.get_dataset_generator(REPRESENTATIONS)
X_data,y_data = dataset_generator.vectorize_dataset()
X_train,  y_train, X_val, y_val, X_holdout, y_holdout = split_dataset(X_data, y_data )

print('X_train.shape={} X_val.shape={} X_holdout.shape={}'.format(X_train.shape, X_val.shape, X_holdout.shape))

gc.collect()

net = None

if REPRESENTATIONS=='word_indeces':
    input_word_dims = 32
    nb_words = dataset_generator.nb_words
    ngram_arity = X_train.shape[1]

    embedding_layer = Embedding(output_dim=input_word_dims,
                                input_dim=nb_words,
                                input_length=ngram_arity,
                                mask_zero=False,
                                trainable=True)

    x_input = Input(shape=(ngram_arity,), dtype='int32')
    emb = embedding_layer(x_input)
    net = Flatten()(emb)

    ndense = ngram_arity*input_word_dims

    net = Dense( units=ndense, activation='relu' )(net)
    net = BatchNormalization()(net)
    net = Dense( units=int(ndense/2), activation='relu' )(net)
    net = BatchNormalization()(net)
    net = Dense( units=int(ndense/3), activation='relu' )(net)
    net = BatchNormalization()(net)
    #net = Dense( units=int(ndense/4), activation='relu' )(net)
    #net = BatchNormalization()(net)

    net = Dense( units=1, activation='sigmoid' )(net)
else:
    input_size = X_train.shape[1]
    x_input = Input(shape=(input_size,), dtype='float32')
    ndense = input_size

    net = Dense( units=ndense, activation='relu' )(x_input)
    net = BatchNormalization()(net)
    net = Dense( units=int(ndense/2), activation='relu' )(net)
    net = BatchNormalization()(net)
    net = Dense( units=int(ndense/3), activation='relu' )(net)
    net = BatchNormalization()(net)
    #net = Dense( units=int(ndense/4), activation='relu' )(net)
    #net = BatchNormalization()(net)
    net = Dense( units=1, activation='sigmoid' )(net)


model = Model( inputs=x_input, outputs=net )
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

weights_filename = 'wr_keras.model'
model_checkpoint = ModelCheckpoint( weights_filename, monitor='val_loss', verbose=1,
                                   save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

history = model.fit(X_data, y_data,
                    batch_size=512,
                    epochs=200,
                    validation_split=0.1,
                    callbacks=[model_checkpoint, early_stopping])

model.load_weights(weights_filename)

# ---------------------------------------------------------------

y_pred = model.predict(X_holdout)
test_loss = sklearn.metrics.log_loss( y_holdout, y_pred )

y_pred  = (y_pred > 0.5).astype(int)
acc = sklearn.metrics.accuracy_score( y_holdout, y_pred )

print('test_loss={} test_acc={}'.format(test_loss, acc))

