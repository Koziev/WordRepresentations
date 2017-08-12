# -*- coding: utf-8 -*-
'''
Головной решатель на базе нейросетки keras для бенчмарка эффективности разных word representation
в задаче определения допустимости N-граммы.
(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import sklearn.model_selection

from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from DatasetVectorizers import WordIndeces_Vectorizer


dataset_generator = WordIndeces_Vectorizer()
X_data,y_data = dataset_generator.vectorize_dataset()

X_train, X_val0, y_train, y_val0 = sklearn.model_selection.train_test_split(X_data, y_data, test_size=0.66,
                                                          random_state=123456)

X_holdout, X_val, y_holdout, y_val = sklearn.model_selection.train_test_split(X_val0, y_val0, test_size=0.50,
                                                          random_state=123456)

print('X_train.shape={} X_val.shape={} X_holdout.shape={}'.format(X_train.shape, X_val.shape, X_holdout.shape))


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
net = Dense( units=int(ndense/2), activation='relu' )(net)
net = Dense( units=1, activation='sigmoid' )(net)

model = Model( inputs=x_input, outputs=net )
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

weights_filename = 'wr_keras.model'
model_checkpoint = ModelCheckpoint( weights_filename, monitor='val_loss', verbose=1,
                                   save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

history = model.fit(X_data, y_data,
                    batch_size=128,
                    epochs=100,
                    validation_split=0.1,
                    callbacks=[model_checkpoint, early_stopping])

model.load_weights(weights_filename)

# ---------------------------------------------------------------

y_pred = model.predict(X_holdout)
test_loss = sklearn.metrics.log_loss( y_holdout, y_pred )

y_pred  = (y_pred > 0.5).astype(int)
acc = sklearn.metrics.accuracy_score( y_holdout, y_pred )

print('test_loss={} test_acc={}'.format(test_loss, acc))

