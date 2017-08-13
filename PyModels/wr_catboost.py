# -*- coding: utf-8 -*-
'''
Головной решатель на базе CatBoost для бенчмарка эффективности разных word representation
в задаче определения допустимости N-граммы.
(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import catboost
import sklearn.model_selection
from DatasetVectorizers import WordIndeces_Vectorizer
from DatasetSplitter import split_dataset

dataset_generator = WordIndeces_Vectorizer()
X_data,y_data = dataset_generator.vectorize_dataset()
X_train,  y_train, X_val, y_val, X_holdout, y_holdout = split_dataset(X_data, y_data )
print('X_train.shape={} X_val.shape={} X_holdout.shape={}'.format(X_train.shape, X_val.shape, X_holdout.shape))


# слова для данного решения представлены просто индексами в словаре, поэтому все столбцы являются
# категориальными признаками для catboost
cat_cols = list(range(X_train.shape[1]))


D_train = catboost.Pool( X_train, label=y_train, cat_features=cat_cols )
D_val = catboost.Pool( X_val, label=y_val, cat_features=cat_cols )

print('Start training...')
model = catboost.CatBoostClassifier(depth=X_train.shape[1],
                           iterations=2000,
                           learning_rate=0.1,
                           eval_metric='Accuracy',
                           random_seed=123456,
                           auto_stop_pval=1e-2,
                                    )

model.fit(D_train, eval_set=D_val, use_best_model=True, verbose=True )

nb_trees = model.get_tree_count()
print('nb_trees={}'.format(nb_trees))

y_pred = model.predict_proba(X_val)[:,1]
test_loss = sklearn.metrics.log_loss( y_holdout, y_pred )

y_pred  = (y_pred > 0.5).astype(int)
acc = sklearn.metrics.accuracy_score( y_holdout, y_pred )

print('test_loss={} test_acc={}'.format(test_loss, acc))

