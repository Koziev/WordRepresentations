# -*- coding: utf-8 -*-
'''
Вспомогательные функции для подготовки датасета для бенчмарка.
(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import sklearn.model_selection


def split_dataset(X_data, y_data):
    X_train, X_val0, y_train, y_val0 = sklearn.model_selection.train_test_split(X_data, y_data, test_size=0.66,
                                                              random_state=123456)

    X_holdout, X_val, y_holdout, y_val = sklearn.model_selection.train_test_split(X_val0, y_val0, test_size=0.50,
                                                              random_state=123456)

    return X_train, y_train, X_val, y_val, X_holdout, y_holdout
