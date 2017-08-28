# -*- coding: utf-8 -*-
'''
Головной решатель на базе XGBoost для бенчмарка эффективности разных word representation
в задаче определения допустимости N-граммы.
(c) Козиев Илья inkoziev@gmail.com

В частности, проверяются подходы с w2v векторами, brown cluster'ами, random projections
и 1-hot кодирование для character surface слов.
'''

from __future__ import print_function
import xgboost
import sklearn.model_selection
from DatasetVectorizers import BaseVectorizer
from DatasetSplitter import split_dataset


# арность N-грамм
NGRAM_ORDER = 2

# кол-во сэмплов в датасете
NB_SAMPLES = 1000000


# проверямеый вариант представления слов в датасете
# w2v - используем word2vec
# w2v_tags - склеиваются w2v векторы слов и дополнительно приклеиваются морфологические признаки каждого слова
# random_bitvector - каждому слову приписывается случайный бинарный вектор фиксированной длины с заданной пропорцией 0/1
# bc - в качестве репрезентаций используются векторы, созданные в результате работы brown clustering
# chars - каждое слово кодируется как цепочка из 1-hot репрезентаций символов
# hashing_trick - используется hashing trick для кодирования слов ограниченным числом битов индекса
# word_freq - единственным признаком слова является его частота в корпусе
REPRESENTATIONS = 'word_freq' # 'w2v' | 'w2v_tags' | 'random_bitvector' | 'bc' | 'chars' | 'hashing_trick' ...


dataset_generator = BaseVectorizer.get_dataset_generator(REPRESENTATIONS)
X_data,y_data = dataset_generator.vectorize_dataset()
X_train,  y_train, X_val, y_val, X_holdout, y_holdout = split_dataset(X_data, y_data )
print('X_train.shape={} X_val.shape={} X_holdout.shape={}'.format(X_train.shape, X_val.shape, X_holdout.shape))

D_train = xgboost.DMatrix(X_train, y_train)
D_val = xgboost.DMatrix(X_val, y_val)
D_holdout = xgboost.DMatrix(X_holdout, y_holdout)


xgb_params = dict()
xgb_params['eta'] = 0.1
xgb_params['max_depth'] = 7
xgb_params['subsample'] = 0.85
xgb_params['min_child_weight'] = 3
xgb_params['gamma'] = 0.05
xgb_params['colsample_bytree'] = 0.85
xgb_params['colsample_bylevel'] = 0.85
xgb_params['objective'] = 'multi:softprob'
xgb_params['seed'] = 123456
xgb_params['silent'] = True
xgb_params['eval_metric'] = 'logloss'
xgb_params['objective'] = 'binary:logistic'

watchlist = [(D_train, 'train'), (D_val, 'valid')]

model = xgboost.train( params=xgb_params,
                      dtrain=D_train,
                      num_boost_round=5000,
                      evals=watchlist,
                      verbose_eval=50,
                      early_stopping_rounds=100)

print('nb_trees={}'.format(model.best_ntree_limit))

y_pred = model.predict(D_holdout, ntree_limit=model.best_ntree_limit )
test_loss = sklearn.metrics.log_loss( y_holdout, y_pred )

y_pred  = (y_pred > 0.5).astype(int)
acc = sklearn.metrics.accuracy_score( y_holdout, y_pred )

print('test_loss={} test_acc={}'.format(test_loss, acc))

