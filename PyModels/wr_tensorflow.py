# -*- coding: utf-8 -*-
'''
Головной решатель на базе нейросетки, созданной средствами TensorFlow,
для бенчмарка эффективности разных word representation в задаче
определения допустимости N-граммы.

(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import numpy as np
import time
import sklearn
import pickle
import tensorflow as tf
import gc
import glob
import os


from DatasetVectorizers import BaseVectorizer
from DatasetSplitter import split_dataset


# арность N-грамм
NGRAM_ORDER = 3

# кол-во сэмплов в датасете
NB_SAMPLES = 1000000

# Выбранный вариант представления слов - см. модуль DatasetVectorizers.py
REPRESENTATIONS = 'w2v' # 'word_indeces' | 'w2v' | 'w2v_tags' | 'char_indeces'

MODEL_DIR = "../data/tf"

# ------------------------------------------------------------------------

def empty_folder(folder):
    files = glob.glob(folder+'/*')
    for f in files:
        if os.path.isdir(f) and f!=folder:
            empty_folder(f)
        elif os.path.isfile(f):
            os.remove(f)

# ------------------------------------------------------------------------

dataset_generator = BaseVectorizer.get_dataset_generator(REPRESENTATIONS)
X_data,y_data = dataset_generator.vectorize_dataset(NGRAM_ORDER, NB_SAMPLES)
gc.collect()

X_train,  y_train, X_val, y_val, X_holdout, y_holdout = split_dataset( X_data, y_data )
del X_data
del y_data
gc.collect()


print('X_train.shape={} X_val.shape={} X_holdout.shape={}'.format(X_train.shape, X_val.shape, X_holdout.shape))


input_size = X_train.shape[1]

# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[input_size])]

tf.logging.set_verbosity(tf.logging.INFO)

empty_folder(MODEL_DIR)

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[input_size, input_size/2],
                                      n_classes=2,
                                      optimizer='Adagrad',
                                      activation_fn=tf.nn.relu,
                                      model_dir=MODEL_DIR,
                                      config=tf.contrib.learn.RunConfig(save_checkpoints_steps=50))

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": X_train},
  y=y_train,
  batch_size=512,
  num_epochs=50,
  num_threads=1,
  shuffle=True)



#validation_monitor = tf.contrib.learn.monitors.ValidationMonitor( X_val, y_val, every_n_steps=50)

# Train model.
classifier.train(input_fn=train_input_fn, steps=2000 )

#classifier.fit(x=X_train, y=y_train, steps=2000, monitors=[validation_monitor])


# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": X_holdout},
  y=y_holdout,
  num_epochs=1,
  shuffle=False)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

