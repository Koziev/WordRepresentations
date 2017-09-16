# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function, unicode_literals)
import sys


class ModelConfigurationSettings(object):

    @staticmethod
    def get_w2v_path():
        if 'linux' in sys.platform.lower():
            return r'/home/eek/polygon/w2v/word_vectors_cbow=1_win=5_dim=32.txt'
        else:
            return r'f:/Word2Vec/word_vectors_cbow=1_win=5_dim=32.txt'

    @staticmethod
    def get_ae_path():
        if 'linux' in sys.platform.lower():
            return r'/home/eek/polygon/WordRepresentations/data/ae_vectors.dat'
        else:
            return r'e:\polygon\WordRepresentations\data\ae_vectors.dat'

