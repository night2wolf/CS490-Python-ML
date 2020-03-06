import nltk
import requests
from bs4 import BeautifulSoup
import urllib.request
import os
from bs4.element import Comment
import urllib.request
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
pstemmer = PorterStemmer()
sbstemmer = SnowballStemmer('english')
lstemmer = LancasterStemmer()
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)
html = urllib.request.urlopen("https://en.wikipedia.org/wiki/Google").read()
bsObj = text_from_html(html)
file = open("input.txt",'w')
data = bsObj.encode("utf-8")
file.write(str(data))
file.close()
inputfile = open("input.txt",'r')
pstemfile = open("pstemmer.txt",'w')
lstemfile = open("lstemmer.txt",'w')
sbstemfile = open("sbstemmer.txt",'w')
postagfile = open("postags.txt",'w')
nertagfile = open("nertag.txt",'w')
tokenize = nltk.word_tokenize(inputfile.read())
inputfile.seek(0)
sent_tokenize = nltk.sent_tokenize(inputfile.read())
sent_tokenize = sent_tokenize[:20]
tokenize = tokenize[:20] 
for w in tokenize:
  print(str(w))
  pstemfile.write(pstemmer.stem(w))
  sbstemfile.write(sbstemmer.stem(w))
  lstemfile.write(lstemmer.stem(w))
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
named = []
for sentence in sent_tokenize:
    ner = ne_chunk(pos_tag(wordpunct_tokenize(sentence)))
    named.append(ner)
    if 0 < len(named) <= 3:
        print(ner)


postagfile.write(str(nltk.pos_tag(tokenize)))
trigrams = list(nltk.trigrams(tokenize))
print(trigrams[:20])

