# -*- coding: utf-8 -*-
'''
Классы для построчного чтения из текстового корпуса.
(c) Козиев Илья inkoziev@gmail.com  https://github.com/Koziev/WordRepresentations
'''

from __future__ import print_function
import codecs
import zipfile


class BaseCorpusReader(object):
    def __init__(self):
        pass

    def get_corpus_info(self):
        raise NotImplementedError


class ZippedCorpusReader(BaseCorpusReader):
    '''
    Корпус упакован в zip-архив для удобства использования проекта с
    git-репозиторием на гитхабе. Предполагается, что в архиве находится
    файл с именем corpus.txt в кодировке utf-8.
    '''
    def __init__(self, zipped_corpus_path):
        self.zipped_corpus_path = zipped_corpus_path

    def _get_corpus_path(self):
        return self.zipped_corpus_path

    def get_corpus_info(self):
        return self._get_corpus_path()

    def read_lines(self):
        with zipfile.ZipFile(self._get_corpus_path()) as z:
            with z.open('corpus.txt') as rdr:
                for line0 in rdr:
                    line = line0.decode('utf-8').strip()
                    yield line



class TxtCorpusReader(BaseCorpusReader):
    '''
    Чтение строк из plain text utf-8 файла
    '''
    def __init__(self, txt_corpus_path):
        self.corpus_path = txt_corpus_path

    def _get_corpus_path(self):
        return self.corpus_path

    def get_corpus_info(self):
        return self._get_corpus_path()()

    def read_lines(self):
        with codecs.open( self._get_corpus_path(), 'r', 'utf-8') as rdr:
            for line0 in rdr:
                line = line0.strip()
                yield line
