# -*- coding: utf-8 -*-
'''
Решение задачи https://github.com/Koziev/WordRepresentations с помощью нейросетки,
созданной средствами TensorFlow high-level api.

Проверяются разные варианты представления слов - см. глобальную переменную REPRESENTATIONS

(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import sklearn
import tensorflow as tf
import gc
import os
from DatasetVectorizers import BaseVectorizer
from DatasetSplitter import split_dataset
import glob

# арность N-грамм
NGRAM_ORDER = 3

# кол-во сэмплов в датасете
NB_SAMPLES = 1000000

# Выбранный вариант представления слов - см. модуль DatasetVectorizers.py
REPRESENTATIONS = 'w2v' #  'w2v' | 'w2v_tags' | 'chars'

BATCH_SIZE = 256

MODEL_DIR = "../data/tf"

# ----------------------------------------------------------------------

input_size = -1

def mlp_model_fn( features, labels, mode):
    net = tf.layers.dense(inputs=features["x"], units=input_size, activation=tf.nn.relu)
    #net = tf.layers.batch_normalization(net)
    net = tf.layers.dense(inputs=net, units=int(input_size/2), activation=tf.nn.relu)
    #net = tf.layers.batch_normalization(net)
    net = tf.layers.dense(inputs=net, units=int(input_size/3), activation=tf.nn.relu)
    #dropout = tf.layers.dropout( inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=net, units=1, activation=tf.sigmoid)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.round( logits ),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
      "probabilities": logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    #loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    #loss = tf.nn.l2_loss( labels-logits )
    #loss = -tf.reduce_sum(labels * tf.log(logits))
    labels1 = tf.reshape(labels,[tf.shape(labels)[0], 1])
    #print('labels1.shape={} logits.shape={}'.format(labels1.get_shape(), logits.get_shape()))
    loss = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits( labels=labels1, logits=logits) )


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.AdamOptimizer()

        train_op = optimizer.minimize( loss=loss, global_step=tf.train.get_global_step() )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# ------------------------------------------------------------------------

def empty_folder(folder):
    files = os.listdir(folder)
    #files = glob.glob(folder+'/*')
    for f in [ os.path.join(folder,f) for f in files]:
        if os.path.isdir(f):
            empty_folder(f)
        elif os.path.isfile(f):
            os.remove(f)


# ----------------------------------------------------------------------

empty_folder(MODEL_DIR)

dataset_generator = BaseVectorizer.get_dataset_generator(REPRESENTATIONS)
X_data,y_data = dataset_generator.vectorize_dataset(NGRAM_ORDER, NB_SAMPLES)
gc.collect()

X_train,  y_train, X_val, y_val, X_holdout, y_holdout = split_dataset( X_data, y_data )

y_train = y_train.astype('float32')
y_val = y_val.astype('float32')
y_holdout = y_holdout.astype('float32')


del X_data
del y_data
gc.collect()


print('X_train.shape={} X_val.shape={} X_holdout.shape={}'.format(X_train.shape, X_val.shape, X_holdout.shape))


input_size = X_train.shape[1]


# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[input_size])]

tf.logging.set_verbosity(tf.logging.INFO)


# Create the Estimator
classifier = tf.estimator.Estimator( model_fn=mlp_model_fn, model_dir=MODEL_DIR )

# Set up logging for predictions
#tensors_to_log = {"probabilities": "softmax_tensor"}
#logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


nb_steps = int(X_train.shape[0]/BATCH_SIZE)
nb_epochs = 20

# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": X_train},
  y=y_train,
  batch_size=BATCH_SIZE,
  num_epochs=nb_epochs,
  num_threads=1,
  shuffle=True)

classifier.train( input_fn=train_input_fn, steps=nb_epochs*nb_steps ) #, hooks=[logging_hook])

# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": X_holdout},
  y=y_holdout,
  num_epochs=1,
  shuffle=False)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

predictions = list(classifier.predict(input_fn=test_input_fn))
predicted_classes = [p["classes"] for p in predictions]

acc = sklearn.metrics.accuracy_score( y_true=y_holdout, y_pred=predicted_classes )
print('prediction acc={}'.format(acc))


print('All done.')
