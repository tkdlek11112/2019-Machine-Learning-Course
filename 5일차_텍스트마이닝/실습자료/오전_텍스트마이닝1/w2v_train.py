# -*- coding: utf-8 -*-
import re
import pandas as pd
import nltk
import multiprocessing
from gensim.models import Word2Vec

df = pd.read_csv('simpsons_dataset.csv')
print(df.shape)
df.head()

# Removes non-alphabetic characters:
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words'])

sentences = []
for idx, s in enumerate(brief_cleaning):
    tokens = nltk.word_tokenize(s)
    tags = nltk.pos_tag(tokens)
    sent = [x for x, _ in tags]
    sentences.append(sent)
    if idx % 100 == 0:
        print('.', end='')

# 1. define model
w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=2)

# 2. make vocab.
w2v_model.build_vocab(sentences, progress_per=10000)

# 3. training
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=50, report_delay=1)
w2v_model.init_sims(replace=True)  # terminate training

# 4. use word2vec
w2v_model.wv.most_similar(positive=["homer"])

w2v_model.wv.most_similar(positive=["simpson"])

w2v_model.wv.most_similar(positive=["marge"])

w2v_model.wv.most_similar(positive=["bart"])
