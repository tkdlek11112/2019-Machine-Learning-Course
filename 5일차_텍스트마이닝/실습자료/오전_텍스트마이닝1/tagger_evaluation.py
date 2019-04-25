import re
import konlpy
from konlpy.tag import Hannanum, Kkma, Komoran, Twitter

test = [
    '격동의 현대사 속에서 수많은 고난과 역경을 극복해 온 우리 앞에 지금 글로벌 경제 위기와 북한의 핵무장 위협과 같은 안보위기가 이어지고 있습니다.',
    '아버지가방에들어가신다',
    '꿀잼 ㅋㅋㅋㅋ'
]

hannanum = Hannanum()
kkma = Kkma()
komoran = Komoran()
twitter = Twitter()

taggers = [hannanum, kkma, komoran, twitter]

for x in test:
    out = [tagger.pos(x, join=True) for tagger in taggers]
    for o in out:
        print(o)

# https://konlpy-ko.readthedocs.io/ko/v0.4.3/morph/#pos-tagging-with-konlpy

