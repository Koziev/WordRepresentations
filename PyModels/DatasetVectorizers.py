# -*- coding: utf-8 -*-
'''
Классы-векторизаторы датасета для программы бенчмаркинга эффективности
разных word representation в задаче определения допустимости N-граммы.
(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import gensim
import numpy as np
import codecs
import collections
import math
import random
from scipy.sparse import lil_matrix
import xgboost
import sklearn.model_selection
from itertools import chain
import sys
import zipfile

# арность N-грамм
NGRAM_ORDER = 3

# кол-во сэмплов в датасете
NB_SAMPLES = 1000000

# -------------------------------------------------------------------

def get_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])


def subclasses(cls):
    """
    Вспомогательная функция для получения всех подклассов для заданного класса,
    включая второй и последующий уровни иерархии, если вдруг понадобиться отнаследоваться
    от Chars_Vectorizer, например.
    :param cls: базовый класс
    :return: список классов-наследников
    """
    return list(
        chain.from_iterable(
            [list(chain.from_iterable([[x], subclasses(x)])) for x in cls.__subclasses__()]
        )
    )


class BaseVectorizer(object):
    """
    Базовый класс для загрузки и векторизации датасета.
    Его задача - вернуть тензоны X_data и y_data, на которых
    будет учиться бинарный классификатор.
    """
    def __init__(self):
        pass

    def _get_corpus_path(self):
        # путь к текстовому корпусу, из которого берутся N-граммы
        corp_path = 'F:/Corpus/word2vector/ru/SENTx.corpus.w2v.txt'
        return corp_path

    def vectorize_dataset(self):
        raise NotImplemented()

    def _generate_dataset(self, all_words, valid_ngrams):
        dataset_x = []
        dataset_y = []

        word_list = list(all_words)

        print('Building the list of {} positive and negative samples'.format(NB_SAMPLES))
        for ngram in valid_ngrams:
            dataset_x.append(ngram)
            dataset_y.append(1)

            # заменяем одно слово на произвольное, получаем (обычно) недопустимое сочетание слов.
            while True:
                iword = random.randint(0, NGRAM_ORDER - 1)
                word = random.choice(word_list)
                n = []

                new_ngram = []

                # слева от заменяемого слова
                if iword > 0:
                    new_ngram.extend( ngram[:iword] )

                # добавляем замену слова
                new_ngram.append(word)

                # справа от заменяемого слова
                if iword < NGRAM_ORDER - 1:
                    new_ngram.extend(ngram[iword + 1 : ])

                new_ngram = tuple(new_ngram)
                if new_ngram not in valid_ngrams:
                    dataset_x.append(new_ngram)
                    dataset_y.append(0)
                    break

        return (dataset_x, dataset_y)

    def _load_ngrams(self, valid_words):
        all_words = set()
        valid_ngrams = set()
        invalid_ngrams = set()

        assert (NB_SAMPLES % 2) == 0
        MAX_NB_1_NGRAMS = NB_SAMPLES / 2

        print('Extracting {}-grams from {}...'.format(NGRAM_ORDER, self._get_corpus_path()))
        with codecs.open(self._get_corpus_path(), "r", "utf-8") as rdr:
            nline = 0
            for line in rdr:
                nline += 1
                if (nline % 10000) == 0:
                    print('{0} lines, {1} ngrams'.format(nline, len(valid_ngrams)), end='\r')

                words = line.strip().split(u' ')

                all_words_known = True
                for word in words:
                    if valid_words is None or word in valid_words:
                        all_words.add(word)

                ngrams = get_ngrams(words, NGRAM_ORDER)
                for ngram in ngrams:
                    if ngram not in valid_ngrams and ngram not in invalid_ngrams:
                        all_words_known = True
                        for word in ngram:
                            if word not in all_words:
                                all_words_known = False
                                break

                        if all_words_known:
                            if len(valid_ngrams) < MAX_NB_1_NGRAMS:
                                valid_ngrams.add(ngram)
                            else:
                                break
                        else:
                            invalid_ngrams.add(ngram)

                if len(valid_ngrams) >= MAX_NB_1_NGRAMS:
                    break

        print(
            'Finished, {0} lines, {1} {2}-grams, {3} words.'.format(nline, len(valid_ngrams), NGRAM_ORDER, len(all_words)))

        (dataset_x,dataset_y) = self._generate_dataset(all_words,valid_ngrams)

        return (all_words, dataset_x, dataset_y)

    @classmethod
    def get_name(self):
        raise NotImplemented()

    @classmethod
    def get_dataset_generator(self, representation_name):
        """
        Фабрика для генераторов датасетов. Класс, который будет выполнять векторизацию, ищется
        по заданной условной строковой метке. Эту же метку возвращает метод get_name() в классах-векторизаторах.

        :param representation_name: наименование способа представления слов
        :return: объект класса, производного от BaseVectorizer, который будет выполнять генерацию матриц датасета.
        """

        for vectorizer_class in subclasses(BaseVectorizer):
            label = getattr(vectorizer_class, 'get_name')()
            if label == representation_name:
                return vectorizer_class()

        raise NotImplemented()

# -------------------------------------------------------------------

class W2V_Vectorizer(BaseVectorizer):
    """
    В качестве репрезентаций слов берем их векторы из ранее обученной word2vec модели.
    Векторы слов в N-грамме склеиваются в один вектор.
    """
    def __init__(self):
        super(W2V_Vectorizer,self).__init__()

    @classmethod
    def get_name(self):
        return 'w2v'

    def _load_w2v(self):
        # путь к word2vec модели
        w2v_path = r'F:\Word2Vec\word_vectors_cbow=1_win=5_dim=32.txt'
        print('Loading w2v model...')
        w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=False)
        return w2v

    def vectorize_dataset(self):
        w2v = self._load_w2v()

        valid_words = set(w2v.vocab)
        (all_words, dataset_x, dataset_y) = self._load_ngrams(valid_words)

        # -------------------------------------------------------------------

        y_data = np.zeros((NB_SAMPLES), dtype='bool')
        y_data[:] = dataset_y[:]

        nword = len(w2v.vocab)
        print('Number of words={0}'.format(nword))
        vec_len = len(w2v.syn0[0])
        print('Vector length={0}'.format(vec_len))

        input_size = vec_len * NGRAM_ORDER
        X_data = np.zeros((NB_SAMPLES, input_size), dtype='float32')

        for idata, ngram in enumerate(dataset_x):
            for iword, word in enumerate(ngram):
                X_data[idata, iword * vec_len:(iword + 1) * vec_len] = w2v[word]

        return (X_data, y_data)

# -------------------------------------------------------------------

class BinaryWord_Vectorizer(BaseVectorizer):
    """
    Самый глупый способ векторизации - просто берем список слов, присваиваем
    каждому слову целочисленный код и вуаля.
    """
    def __init__(self):
        super(BinaryWord_Vectorizer,self).__init__()

    @classmethod
    def get_name(self):
        return 'random_bitvector'

    def vectorize_dataset(self):

        (all_words, dataset_x, dataset_y) = self._load_ngrams(valid_words=None)

        nb_words = len(all_words)

        print('Generating {} word vectors...'.format(nb_words) )
        vec_len = 128  # длина битового вектора для каждого слова
        nb_1s = 16 # сколько единичных битов будет в каждом векторе
        word2index = dict([ (w,i) for i,w in enumerate(all_words)])

        p1 = float(nb_1s)/vec_len
        p0 = float(vec_len-nb_1s)/vec_len

        word_vectors = np.zeros( (nb_words,vec_len), dtype='bool' )
        for iword,word in enumerate(all_words):
            v = np.random.choice([0, 1], size=(vec_len,), p=[p0,p1])
            word_vectors[iword] = v

        # -------------------------------------------------------------------

        y_data = np.zeros((NB_SAMPLES), dtype='bool')
        y_data[:] = dataset_y[:]

        input_size = vec_len * NGRAM_ORDER
        X_data = np.zeros((NB_SAMPLES, input_size), dtype='bool')

        for idata, ngram in enumerate(dataset_x):
            for iword, word in enumerate(ngram):
                X_data[idata, iword * vec_len:(iword + 1) * vec_len] = word_vectors[ word2index[word] ]

        return (X_data, y_data)

# -------------------------------------------------------------------

class BrownClusters_Vectorizer(BaseVectorizer):
    """
    В качестве репрезентаций слов берем результаты brown clustering (https://en.wikipedia.org/wiki/Brown_clustering).
    """
    def __init__(self):
        super(BrownClusters_Vectorizer,self).__init__()

    @classmethod
    def get_name(self):
        return 'bc'

    def _load_bc(self):
        # путь к word2vec модели
        bc_path = r'../data/paths'
        print('Loading brown clusters from {}...'.format(bc_path))

        w2bc = dict()
        with codecs.open(bc_path, "r", "utf-8") as rdr:
            for line in rdr:
                tx = line.strip().split(u'\t')
                c = tx[0]
                word = tx[1]
                w2bc[word] = c

        return w2bc

    def vectorize_dataset(self):

        w2bc = self._load_bc()

        valid_words = set( w2bc.keys() )
        (all_words, dataset_x, dataset_y) = self._load_ngrams(valid_words)

        vec_len = max( len(c) for c in w2bc.values() )
        print('Vector length={0}'.format(vec_len))

        # -------------------------------------------------------------------

        print('Vectorize {} samples in dataset...'.format( len(dataset_x) ) )
        y_data = np.zeros((NB_SAMPLES), dtype='bool')
        y_data[:] = dataset_y[:]

        input_size = vec_len * NGRAM_ORDER
        X_data = lil_matrix( (NB_SAMPLES, input_size), dtype='bool')

        for idata, ngram in enumerate(dataset_x):
            for iword, word in enumerate(ngram):
                v0 = w2bc[word]
                v = v0.ljust(vec_len,' ')
                for j,x in enumerate(v):
                    if x=='0' or x=='1':
                        X_data[idata, iword * vec_len + j ] = (x=='1')

        return (X_data, y_data)

# -------------------------------------------------------------------

class Chars_Vectorizer(BaseVectorizer):
    """
    1-hot encoding для символов.
    Репрезентации слов получаются склеиванием векторов символов.
    """
    def __init__(self):
        super(Chars_Vectorizer,self).__init__()

    @classmethod
    def get_name(self):
        return 'chars'

    def vectorize_dataset(self):
        (all_words, dataset_x, dataset_y) = self._load_ngrams(valid_words=None)

        nb_words = len(all_words)

        all_chars = set()
        for w in all_words:
            all_chars.update(w)

        nb_chars = len(all_chars)
        char2index = dict( [ (c,i) for i,c in enumerate(all_chars)] )


        max_word_len = max( [len(w) for w in all_words ] )

        vec_len = nb_chars*max_word_len  # длина битового вектора для каждого слова

        # -------------------------------------------------------------------

        y_data = np.zeros((NB_SAMPLES), dtype='bool')
        y_data[:] = dataset_y[:]

        input_size = vec_len * NGRAM_ORDER
        X_data = lil_matrix( (NB_SAMPLES, input_size), dtype='bool')

        for idata, ngram in enumerate(dataset_x):
            for iword, word in enumerate(ngram):
                for j,c in enumerate(word[::-1]):
                    X_data[idata, iword * vec_len + j*nb_chars + char2index[c]] = True

        return (X_data, y_data)

# -------------------------------------------------------------------

class HashingTrick_Vectorizer(BaseVectorizer):
    """
    Использование Hashing Trick (https://en.wikipedia.org/wiki/Feature_hashing)
    """
    def __init__(self):
        super(HashingTrick_Vectorizer,self).__init__()

    @classmethod
    def get_name(self):
        return 'hashing_trick'

    def vectorize_dataset(self):

        (all_words, dataset_x, dataset_y) = self._load_ngrams(valid_words=None)

        NB_SLOTS = 32000 # столько элементов будет в хэш-таблице, так что
                         # некоторое количество слов будут давать коллизии
        hash_dict = gensim.corpora.hashdictionary.HashDictionary( id_range=NB_SLOTS, debug=True)

        nb_words = len(all_words)

        # -------------------------------------------------------------------

        y_data = np.zeros((NB_SAMPLES), dtype='bool')
        y_data[:] = dataset_y[:]

        input_size = NB_SLOTS * NGRAM_ORDER
        X_data = lil_matrix( (NB_SAMPLES, input_size), dtype='bool')

        for idata, ngram in enumerate(dataset_x):
            tokens = hash_dict.doc2bow( ngram, allow_update=True )
            for iword,(token_id,token_count) in enumerate(tokens):
                X_data[idata, iword*NB_SLOTS + token_id] = True

        return (X_data, y_data)

# -------------------------------------------------------------------

class WordIndeces_Vectorizer(BaseVectorizer):
    """
    В качестве репрезентаций слов берем их индексы (случайное упорядочивание)
    """
    def __init__(self):
        super(WordIndeces_Vectorizer,self).__init__()
        self.all_words = []

    @classmethod
    def get_name(self):
        return 'word_indeces'

    @property
    def nb_words(self):
        return len(self.all_words)

    def vectorize_dataset(self):

        (self.all_words, dataset_x, dataset_y) = self._load_ngrams(None)
        word2id = dict([(w,i) for i,w in enumerate(self.all_words)])

        # -------------------------------------------------------------------

        y_data = np.zeros((NB_SAMPLES), dtype='bool')
        y_data[:] = dataset_y[:]

        input_size = NGRAM_ORDER
        X_data = np.zeros((NB_SAMPLES, input_size), dtype='int32')

        for idata, ngram in enumerate(dataset_x):
            for iword, word in enumerate(ngram):
                X_data[idata, iword ] = word2id[word]

        return (X_data, y_data)

# -------------------------------------------------------------------

class W2V_Tags_Vectorizer(W2V_Vectorizer):
    """
    В качестве репрезентаций слов берем их векторы из ранее обученной word2vec модели
    и добавляем морфологические признаки из грамматического словаря.
    """
    def __init__(self):
        super(W2V_Tags_Vectorizer,self).__init__()

    @classmethod
    def get_name(self):
        return 'w2v_tags'

    def vectorize_dataset(self):

        # загружаем грамматический словарь
        tag2id = dict()
        tagset2id = dict()
        id2tagset = dict()
        word2tagset = dict()
        tagged_words = set()

        with zipfile.ZipFile('../data/word2tags_bin.zip') as z:
            with z.open('word2tags_bin.dat') as rdr:
                for line in rdr:
                    tx = line.decode('utf-8').strip().split(u'\t')
                    word = tx[0]
                    tagset = []
                    for tag in tx[1:]:
                        if tag not in tag2id:
                            tag2id[tag] = len(tag2id)
                        tagset.append(tag2id[tag])

                    tagset = tuple(tagset)
                    if tagset not in tagset2id:
                        t_id = len(tagset2id)
                        tagset2id[tagset] = t_id
                        id2tagset[t_id] = tagset
                    word2tagset[word] = tagset2id[tagset]
                    tagged_words.add(word)


        # with codecs.open('word2tags_bin.dat','r','utf-8') as rdr:
        #     for line in rdr:
        #         tx = line.strip().split(u'\t')
        #         word = tx[0]
        #         tagset = []
        #         for tag in tx[1:]:
        #             if tag not in tag2id:
        #                 tag2id[tag] = len(tag2id)
        #             tagset.append(tag2id[tag])
        #
        #         tagset = tuple(tagset)
        #         if tagset not in tagset2id:
        #             t_id = len(tagset2id)
        #             tagset2id[tagset] = t_id
        #             id2tagset[t_id] = tagset
        #         word2tagset[word] = tagset2id[tagset]
        #         tagged_words.add(word)



        w2v = self._load_w2v()

        valid_words = set(w2v.vocab) & tagged_words

        (all_words, dataset_x, dataset_y) = self._load_ngrams(valid_words)

        # -------------------------------------------------------------------

        y_data = np.zeros((NB_SAMPLES), dtype='bool')
        y_data[:] = dataset_y[:]

        nword = len(w2v.vocab)
        print('Number of words={0}'.format(nword))
        vec_len = len(w2v.syn0[0])
        print('Vector length={0}'.format(vec_len))

        tags_per_word = len(tag2id)
        input_size = vec_len * NGRAM_ORDER + tags_per_word*NGRAM_ORDER
        X_data = np.zeros((NB_SAMPLES, input_size), dtype='float32')

        for idata, ngram in enumerate(dataset_x):
            # склеиваем w2v векторы слов
            for iword, word in enumerate(ngram):
                X_data[idata, iword * vec_len:(iword + 1) * vec_len] = w2v[word]
            # справа приклеим еще морфологические признаки
            i0 = len(ngram)*vec_len
            for iword, word in enumerate(ngram):
                tagset = id2tagset[word2tagset[word]]
                for tag_id in tagset:
                    X_data[idata, i0 + iword*tags_per_word+ tag_id] = 1.0

        return (X_data, y_data)

