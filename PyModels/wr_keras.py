# -*- coding: utf-8 -*-
'''
Головной решатель на базе нейросетки keras для бенчмарка эффективности разных word representation
в задаче определения допустимости N-граммы.

Реализованы следующие архитектуры нейросети:
MLP - простая feedforward сетка
ConvNet - сверточные слои

(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import gc
import sklearn
#import sklearn.model_selection
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D
from DatasetVectorizers import BaseVectorizer
from DatasetSplitter import split_dataset



REPRESENTATIONS = 'char_indeces' # 'word_indeces' | 'w2v' | 'w2v_tags' | 'char_indeces'
NET_ARCH = 'MLP' # 'MLP' | 'ConvNet'

# -----------------------------------------------------------------------

def build_model( dataset_generator, X_data ):
    ngram_arity = dataset_generator.get_ngram_arity()

    use_conv = NET_ARCH == 'ConvNet'
    net = None

    if REPRESENTATIONS == 'word_indeces':
        input_word_dims = 32
        nb_words = dataset_generator.nb_words
        ndense = ngram_arity * input_word_dims

        embedding_layer = Embedding(output_dim=input_word_dims,
                                    input_dim=nb_words,
                                    input_length=ngram_arity,
                                    mask_zero=False,
                                    trainable=True)

        x_input = Input(shape=(ngram_arity,), dtype='int32')
        emb = embedding_layer(x_input)

        if use_conv:
            print('Building ConvNet...')
            nets = []

            conv_layer0 = Conv1D(filters=input_word_dims,
                                 kernel_size=1,
                                 padding='valid',
                                 activation='relu',
                                 strides=1)
            net0 = conv_layer0(emb)
            net0 = GlobalMaxPooling1D()(net0)
            nets.append(net0)

            conv_layer1 = Conv1D(filters=input_word_dims,
                                 kernel_size=2,
                                 padding='valid',
                                 activation='relu',
                                 strides=1)
            net1 = conv_layer1(emb)
            net1 = GlobalMaxPooling1D()(net1)
            nets.append(net1)

            if ngram_arity >= 3:
                conv_layer2 = Conv1D(filters=input_word_dims,
                                     kernel_size=3,
                                     padding='valid',
                                     activation='relu',
                                     strides=1)

                net2 = conv_layer2(emb)
                net2 = GlobalMaxPooling1D()(net2)
                nets.append(net2)

            if ngram_arity >= 4:
                conv_layer3 = Conv1D(filters=input_word_dims,
                                     kernel_size=4,
                                     padding='valid',
                                     activation='relu',
                                     strides=1)

                net3 = conv_layer3(emb)
                net3 = GlobalMaxPooling1D()(net3)
                nets.append(net3)

            net = concatenate(nets)

            net = Dense(units=int(input_word_dims / 2), activation='relu')(net)
            net = BatchNormalization()(net)
            net = Dense(units=int(input_word_dims / 3), activation='relu')(net)
            net = BatchNormalization()(net)
            net = Dense(units=1, activation='sigmoid')(net)

        else:
            print('Building MLP...')
            net = Flatten()(emb)
            net = Dense(units=ndense, activation='relu')(net)
            net = BatchNormalization()(net)
            net = Dense(units=int(ndense / 2), activation='relu')(net)
            net = BatchNormalization()(net)
            net = Dense(units=int(ndense / 3), activation='relu')(net)
            net = BatchNormalization()(net)
            # net = Dense( units=int(ndense/4), activation='relu' )(net)
            # net = BatchNormalization()(net)
            net = Dense(units=1, activation='sigmoid')(net)

    elif REPRESENTATIONS == 'char_indeces':
        input_char_dims = 16
        nb_chars = dataset_generator.nb_chars
        input_seq_len = X_data.shape[1]
        # ndense = ngram_arity*input_word_dims

        embedding_layer = Embedding(output_dim=input_char_dims,
                                    input_dim=nb_chars,
                                    input_length=input_seq_len,
                                    mask_zero=False,
                                    trainable=True)

        x_input = Input(shape=(input_seq_len,), dtype='int32')
        emb = embedding_layer(x_input)

        if use_conv:
            print('Building ConvNet...')
            nets = []

            conv_layer1 = Conv1D(filters=32,
                                 kernel_size=2,
                                 padding='valid',
                                 activation='relu',
                                 strides=1)
            net1 = conv_layer1(emb)
            net1 = GlobalMaxPooling1D()(net1)
            nets.append(net1)

            if ngram_arity >= 3:
                conv_layer2 = Conv1D(filters=32,
                                     kernel_size=3,
                                     padding='valid',
                                     activation='relu',
                                     strides=1)

                net2 = conv_layer2(emb)
                net2 = GlobalMaxPooling1D()(net2)
                nets.append(net2)

            if ngram_arity >= 4:
                conv_layer3 = Conv1D(filters=32,
                                     kernel_size=4,
                                     padding='valid',
                                     activation='relu',
                                     strides=1)

                net3 = conv_layer3(emb)
                net3 = GlobalMaxPooling1D()(net3)
                nets.append(net3)

            net = concatenate(nets)

            net = Dense(units=int(input_seq_len / 2), activation='relu')(net)
            net = BatchNormalization()(net)
            net = Dense(units=int(input_seq_len / 3), activation='relu')(net)
            net = BatchNormalization()(net)
            net = Dense(units=1, activation='sigmoid')(net)

        else:
            print('Building MLP...')
            ndense = input_seq_len * input_char_dims
            net = Flatten()(emb)
            net = Dense(units=ndense, activation='relu')(net)
            net = BatchNormalization()(net)
            net = Dense(units=int(ndense / 2), activation='relu')(net)
            net = BatchNormalization()(net)
            net = Dense(units=int(ndense / 3), activation='relu')(net)
            net = BatchNormalization()(net)
            # net = Dense( units=int(ndense/4), activation='relu' )(net)
            # net = BatchNormalization()(net)
            net = Dense(units=1, activation='sigmoid')(net)

    else:
        input_size = X_train.shape[1]
        x_input = Input(shape=(input_size,), dtype='float32')
        ndense = input_size

        if use_conv:
            print('Building ConvNet...')

            input_word_dims = X_train.shape[1] / ngram_arity
            net = Reshape((ngram_arity, input_word_dims))(x_input)

            nets = []

            conv_layer0 = Conv1D(filters=input_word_dims,
                                 kernel_size=1,
                                 padding='valid',
                                 activation='relu',
                                 strides=1)
            net0 = conv_layer0(net)
            net0 = GlobalMaxPooling1D()(net0)
            nets.append(net0)

            conv_layer1 = Conv1D(filters=input_word_dims,
                                 kernel_size=2,
                                 padding='valid',
                                 activation='relu',
                                 strides=1)
            net1 = conv_layer1(net)
            net1 = GlobalMaxPooling1D()(net1)
            nets.append(net1)

            if ngram_arity >= 3:
                conv_layer2 = Conv1D(filters=input_word_dims,
                                     kernel_size=3,
                                     padding='valid',
                                     activation='relu',
                                     strides=1)

                net2 = conv_layer2(net)
                net2 = GlobalMaxPooling1D()(net2)
                nets.append(net2)

            if ngram_arity >= 4:
                conv_layer3 = Conv1D(filters=input_word_dims,
                                     kernel_size=4,
                                     padding='valid',
                                     activation='relu',
                                     strides=1)

                net3 = conv_layer3(net)
                net3 = GlobalMaxPooling1D()(net3)
                nets.append(net3)

            net = concatenate(nets)

            net = Dense(units=int(input_word_dims / 2), activation='relu')(net)
            net = BatchNormalization()(net)
            net = Dense(units=int(input_word_dims / 3), activation='relu')(net)
            net = BatchNormalization()(net)
            net = Dense(units=1, activation='sigmoid')(net)

        else:
            print('Building MLP...')
            net = Dense(units=ndense, activation='relu')(x_input)
            net = BatchNormalization()(net)
            net = Dense(units=int(ndense / 2), activation='relu')(net)
            net = BatchNormalization()(net)
            net = Dense(units=int(ndense / 3), activation='relu')(net)
            net = BatchNormalization()(net)
            # net = Dense( units=int(ndense/4), activation='relu' )(net)
            # net = BatchNormalization()(net)
            net = Dense(units=1, activation='sigmoid')(net)

    model = Model(inputs=x_input, outputs=net)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


# -----------------------------------------------------------------------

# Загружаем датасет
# TODO: в будущем надо перейти к работе через итераторы для минибатчей
dataset_generator = BaseVectorizer.get_dataset_generator(REPRESENTATIONS)
X_data,y_data = dataset_generator.vectorize_dataset()
gc.collect()
X_train,  y_train, X_val, y_val, X_holdout, y_holdout = split_dataset(X_data, y_data )
ngram_arity = dataset_generator.get_ngram_arity()

print('X_train.shape={} X_val.shape={} X_holdout.shape={}'.format(X_train.shape, X_val.shape, X_holdout.shape))
gc.collect()

# Создаем сетку нужной архитектуры
model = build_model(dataset_generator, X_data)

weights_filename = 'wr_keras.model'
model_checkpoint = ModelCheckpoint( weights_filename, monitor='val_loss', verbose=1,
                                   save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

# Обучаем на датасете
print('Train...')
history = model.fit(X_data, y_data,
                    batch_size=512,
                    epochs=200,
                    validation_split=0.1,
                    callbacks=[model_checkpoint, early_stopping])

model.load_weights(weights_filename)

# Оценим точность обученной модели
y_pred = model.predict(X_holdout)
test_loss = sklearn.metrics.log_loss( y_holdout, y_pred )

y_pred  = (y_pred > 0.5).astype(int)
acc = sklearn.metrics.accuracy_score( y_holdout, y_pred )

print('test_loss={} test_acc={}'.format(test_loss, acc))

