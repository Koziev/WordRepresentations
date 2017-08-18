# -*- coding: utf-8 -*-
'''
Подготовка к генерации SDR слов.
Загружаем w2v модель из бинарного/текстового файла и генерируем
текстовый файл для утилиты спарсификации-бинаризации
'''
from __future__ import print_function
import gensim
import sys
import codecs
import collections


NWORD = 500000 # сохраним векторы для NWORD самых частотных слов
CORPUS_FILE = '/home/eek/Corpus/word2vector/ru/SENTx.corpus.w2v.txt'
#W2V_FILE = '/home/eek/polygon/w2v/w2v.CBOW=0_WIN=1_DIM=256.model'
W2V_FILE = '/home/eek/polygon/w2v/word_vectors_cbow=1_win=5_dim=32.txt'


# Нам нужно получить частоты слов в исходном корпусе, чтобы эффективно
# взять TOP N.
print( 'Frequency analysis of corpus {}...'.format(CORPUS_FILE) ) 
MAX_NB_LINES = 10000000
word2freq = collections.Counter()
line_count = 0
with codecs.open( CORPUS_FILE, 'r', 'utf-8' ) as rdr:
    for line in rdr:
        line = line.strip()
        line_count += 1
        words = line.split(u' ')
        for word in words:
            word2freq[word] += 1

        if line_count>MAX_NB_LINES:
            print('{} lines, {} words in lexicon'.format(line_count, len(word2freq)))
            break
        elif (line_count%1000)==0:
            print('{} lines, {} words in lexicon'.format(line_count, len(word2freq)), end='\r')

print( 'Total vocabulary size=', len(word2freq) )
words = word2freq.most_common(NWORD)
print( 'Truncated vocabulary size=', len(words) )

print( 'Top 10 words:' )
for i in range(10):
    print( words[i][0], words[i][1] )

# сохраним список слов в текстовом файле
print( 'Writing dict.txt' )
with codecs.open( 'dict.txt', 'w', 'utf-8' ) as wrt:
    for i,w in enumerate(words):
        wrt.write( u'{}\t{}\t{}\n'.format( i, w[0], w[1] ) )

          
print( 'Loading the w2v model {}...'.format(W2V_FILE) )
w2v = gensim.models.KeyedVectors.load_word2vec_format(W2V_FILE, binary=False)

print('Storing output data...')
with codecs.open( 'input_vectors.txt', 'w', 'utf-8' ) as wrt:
    for (word,freq) in words:
        if word not in w2v:
            print( u'missing word: {} freq={}'.format( word, freq ) )
            #sys.exit()
        else:
            wrt.write( u'{} {}\n'.format( word, " ".join([ str(x) for x in w2v[word] ]) ) )

