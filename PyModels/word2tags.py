# -*- coding: utf-8 -*-
'''
Конвертация грамматического словаря в более компактную форму с числовым кодированием
морфологических тегов.
(c) Козиев Илья inkoziev@gmail.com
'''

from __future__ import print_function
import codecs


tag2id = dict()
with codecs.open('../data/word2tags_bin.dat','w','utf-8') as wrt:
    with codecs.open('../data/word2tags.dat','r','utf_8_sig') as rdr:
        for line in rdr:
            parts = line.strip().split(u'\t')
            word = parts[0].lower()
            wrt.write(u'{}'.format(word))

            pos = parts[1]
            if pos not in tag2id:
                tag2id[pos] = len(tag2id)
            wrt.write(u'\t{}'.format(tag2id[pos]))

            tags = parts[2].split(u' ') if len(parts)==3 else []
            for tag in tags:
                if tag not in tag2id:
                    tag2id[tag] = len(tag2id)

                wrt.write(u'\t{}'.format(tag2id[tag]))
            wrt.write(u'\n')
