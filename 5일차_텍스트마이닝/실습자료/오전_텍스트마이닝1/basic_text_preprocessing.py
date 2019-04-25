# -*- coding: utf-8 -*-
import re
import konlpy
from konlpy.tag import Hannanum, Kkma, Komoran
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# data load
def read_speech(path):
    with open(path, 'rb') as f:
        speech = ""
        while True:
            line = f.readline().decode('utf-8')
            speech += line
            if not line:
                break
    return speech


# sentence split
def sentence_split(x):
    x = re.sub("\n+", ' ', x)
    x = re.sub("\s+", ' ', x)
    x = x.split('. ')
    x = [s + '.' for s in x]
    x[-1] = x[-1][:-1]
    return x


# extract nouns
def extract_nouns(inputs):
    nouns = [han.nouns(x) for x in inputs]
    nouns = sum(nouns, [])
    nouns = [n for n in nouns if len(n) > 1]
    return nouns


# load speech president Park & Moon
s_park = read_speech("Z:/1. 프로젝트/2019_한화시스템교육/교육자료_ML/5일차_텍스트마이닝/TM1/examples/speech_Park.txt")
s_moon = read_speech("Z:/1. 프로젝트/2019_한화시스템교육/교육자료_ML/5일차_텍스트마이닝/TM1/examples/speech_Moon.txt")

# step1: sentence split
s_park = sentence_split(s_park)
s_moon = sentence_split(s_moon)

# step2: tokenization
# skip

# step3: POS tagging
han = Hannanum()
kko = Kkma()
komo = Komoran()

han.pos(s_park[0])
han.nouns(s_park[0])


# step4: plotting
# word cloud for Park
s_park_noun = extract_nouns(s_park)
count = Counter(s_park_noun)
tags = count.most_common(100)
# WordCloud, matplotlib: 단어 구름 그리기
font_path = "C:/WINDOWS/Fonts/NANUMGOTHIC.TTF"
wc = WordCloud(font_path=font_path, background_color='white', width=800, height=600)
cloud = wc.generate_from_frequencies(dict(tags))
plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(cloud)
plt.savefig('park.png', dpi=600)
plt.show()


# word cloud for Moon
s_moon_noun = extract_nouns(s_moon)
count = Counter(s_moon_noun)
tags = count.most_common(100)
# WordCloud, matplotlib: 단어 구름 그리기
font_path = "C:/WINDOWS/Fonts/NANUMGOTHIC.TTF"
wc = WordCloud(font_path=font_path, background_color='white', width=800, height=600)
cloud = wc.generate_from_frequencies(dict(tags))
plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(cloud)
plt.savefig('moon.png', dpi=600)
plt.show()