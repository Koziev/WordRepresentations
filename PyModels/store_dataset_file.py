# -*- coding: utf-8 -*-
'''
Генерация датасета и сохранение его в файлах, чтобы запускать модели, написанные на других ЯП.
(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import sklearn.model_selection
import numpy as np
from DatasetVectorizers import W2V_Vectorizer



dataset_generator = W2V_Vectorizer()
X_data,y_data = dataset_generator.vectorize_dataset()

X_train, X_val0, y_train, y_val0 = sklearn.model_selection.train_test_split(X_data, y_data, test_size=0.66,
                                                          random_state=123456)

X_holdout, X_val, y_holdout, y_val = sklearn.model_selection.train_test_split(X_val0, y_val0, test_size=0.50,
                                                          random_state=123456)

print('X_train.shape={} X_val.shape={} X_holdout.shape={}'.format(X_train.shape, X_val.shape, X_holdout.shape))

np.savetxt( '../data/X_train.csv', X_train, fmt='%.18e', delimiter='\t')
np.savetxt( '../data/y_train.csv', y_train, fmt='%.18e', delimiter='\t')

np.savetxt( '../data/X_val.csv', X_val, fmt='%.18e', delimiter='\t')
np.savetxt( '../data/y_val.csv', y_val, fmt='%.18e', delimiter='\t')

np.savetxt( '../data/X_holdout.csv', X_holdout, fmt='%.18e', delimiter='\t')
np.savetxt( '../data/y_holdout.csv', y_holdout, fmt='%.18e', delimiter='\t')
