# -*- coding: utf-8 -*-
'''
Загрузчик грамматического словаря для использования в моделях автоэнкодера.

(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import zipfile
import codecs
import numpy as np
import sys

class GrammarDictionary(object):
    def __init__(self):
        if 'linux' in sys.platform.lower():
            self.tags_path = '/home/eek/polygon/WordRepresentations/data/word2tags_bin.zip'
            self.word2lemma_path = '/home/eek/polygon/WordRepresentations/data/word2lemma.dat'
            self.w2v_path = r'/home/eek/polygon/w2v/word_vectors_cbow=1_win=5_dim=32.txt'
        else:
            self.tags_path = 'e:/polygon/WordRepresentations/data/word2tags_bin.zip'
            self.word2lemma_path = 'e:/polygon/WordRepresentations/data/word2lemma.dat'
            self.w2v_path = r'f:/Word2Vec/word_vectors_cbow=1_win=5_dim=32.txt'


    def load(self, need_w2v=None):
        '''
        загружаем грамматический словарь в память
        '''

        self.w2v = dict()
        self.w2v_veclen = -1
        if need_w2v is not None and need_w2v==True:
            with codecs.open(self.w2v_path, 'r', 'utf-8') as rdr:
                for line in rdr:
                    if self.w2v_veclen == -1:
                        tx = line.strip().split()
                        nb_words = int(tx[0])
                        self.w2v_veclen = int(tx[1])
                    else:
                        tx = line.strip().split()
                        word = tx[0]
                        vec = [float(z) for z in tx[1:]]
                        vec = np.asarray(vec, dtype='float32')
                        self.w2v[word] = vec

        self.word2lemma = dict()
        with codecs.open(self.word2lemma_path, 'r', 'utf-8') as rdr:
            for line in rdr:
                tx = line.strip().split(u'\t')
                word = tx[0].lower()
                lemma = tx[1].lower()
                self.word2lemma[word] = lemma


        self.tag2id = dict()
        self.tagset2id = dict()
        self.id2tagset = dict()
        self.word2tagset = dict()
        self.tagged_words = set()

        with zipfile.ZipFile(self.tags_path) as z:
            with z.open('word2tags_bin.dat') as rdr:
                for line in rdr:
                    tx = line.decode('utf-8').strip().split(u'\t')
                    word = tx[0]
                    tagset = []
                    for tag in tx[1:]:
                        if tag not in self.tag2id:
                            self.tag2id[tag] = len(self.tag2id)
                        tagset.append(self.tag2id[tag])

                    tagset = tuple(tagset)
                    if tagset not in self.tagset2id:
                        t_id = len(self.tagset2id)
                        self.tagset2id[tagset] = t_id
                        self.id2tagset[t_id] = tagset
                    self.word2tagset[word] = self.tagset2id[tagset]
                    self.tagged_words.add(word)

        self.tags_size = max( self.tag2id.values() )+1

        self.known_words = self.tagged_words & set(self.word2lemma.keys())
        if len(self.w2v)>0:
            self.known_words = self.known_words & set(self.w2v.keys())


    def __contains__(self, word):
        return word in self.known_words

    def get_tags_size(self):
        return self.tags_size

    def get_w2v_size(self):
        return self.w2v_veclen

    def get_tags(self, word):
        return self.id2tagset[ self.word2tagset[word] ]

    def get_lemma(self, word):
        return self.word2lemma[word]

    def get_w2v(self, word):
        return self.w2v[word]

    @property
    def vocab(self):
        return self.known_words
