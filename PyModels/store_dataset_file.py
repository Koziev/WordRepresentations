# -*- coding: utf-8 -*-
'''
Генерация датасета и сохранение его в файлах, чтобы запускать модели, написанные на других ЯП.
(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import sklearn.model_selection
import numpy as np
from DatasetVectorizers import W2V_Vectorizer
from DatasetVectorizers import WordIndeces_Vectorizer



dataset_generator = W2V_Vectorizer()
#dataset_generator = WordIndeces_Vectorizer()
X_data,y_data = dataset_generator.vectorize_dataset()

X_train, X_val0, y_train, y_val0 = sklearn.model_selection.train_test_split(X_data, y_data, test_size=0.66,
                                                          random_state=123456)

X_holdout, X_val, y_holdout, y_val = sklearn.model_selection.train_test_split(X_val0, y_val0, test_size=0.50,
                                                          random_state=123456)

print('X_train.shape={} X_val.shape={} X_holdout.shape={}'.format(X_train.shape, X_val.shape, X_holdout.shape))


# --------------------------------------------------------------------
# Датасеты для OpenNN, в которых X и y совмещены в одном файле
nx = X_train.shape[1]
nrow = 10000000
x_header = str.join( '\t', [ 'x{}'.format(i) for i in range(nx) ] )+'\ty'
Xy_train = np.hstack( (X_train, y_train.reshape( (y_train.shape[0],1) ) ) )[:nrow,:]
np.savetxt( '../data/Xy_train.csv', Xy_train, fmt='%.18e', delimiter='\t', header=x_header, comments='')

Xy_val = np.hstack( (X_val, y_val.reshape( (y_val.shape[0],1) ) ) )[:nrow,:]
np.savetxt( '../data/Xy_val.csv', Xy_val, fmt='%.18e', delimiter='\t', header=x_header, comments='')

Xy_holdout = np.hstack( (X_holdout, y_holdout.reshape( (y_holdout.shape[0],1) ) ) )[:nrow,:]
np.savetxt( '../data/Xy_holdout.csv', Xy_holdout, fmt='%.18e', delimiter='\t', header=x_header, comments='')

# --------------------------------------------------------------------

np.savetxt( '../data/X_train.csv', X_train, fmt='%.18e', delimiter='\t')
np.savetxt( '../data/y_train.csv', y_train, fmt='%.18e', delimiter='\t')

np.savetxt( '../data/X_val.csv', X_val, fmt='%.18e', delimiter='\t')
np.savetxt( '../data/y_val.csv', y_val, fmt='%.18e', delimiter='\t')

np.savetxt( '../data/X_holdout.csv', X_holdout, fmt='%.18e', delimiter='\t')
np.savetxt( '../data/y_holdout.csv', y_holdout, fmt='%.18e', delimiter='\t')
