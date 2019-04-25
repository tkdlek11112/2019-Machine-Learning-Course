# -*- coding: utf-8 -*-
from gensim.models import KeyedVectors
# Load vectors directly from the file
model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
# Access vectors for specific words with a keyed lookup:
vector = model['easy']
# see the shape of the vector (300,)
vector.shape
# Processing sentences is not as simple as with Spacy:
vectors = [model[x] for x in "This is some text I am processing with Spacy".split(' ')]

model.wv.most_similar('dog')

model.wv.doesnt_match(['man', 'king', 'woman'])

model.wv.most_similar(positive=['woman', 'king'], negative=['man'])


