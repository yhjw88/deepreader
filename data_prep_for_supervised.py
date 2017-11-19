import requests
from bs4 import BeautifulSoup
import numpy as np

def getArticleNYT(url):
    session = requests.Session()
    req = session.get(url)
    soup = BeautifulSoup(req.text,"lxml")
    title=soup.title.string
    paragraphs = soup.find_all('p', class_='story-body-text story-content')
    article=[]
    for p in paragraphs:
        article.append(p.get_text())
    article=' '.join(article)
    return title,article

#pairing (wiki_id,url)
relepairs=[]
for i in range(len(d)):
    for j in range(len(d[i]["references"])):
        if "url" in d[i]["references"][j]:
            address=d[i]["references"][j]["url"]
            if "nytimes" in address:
                entry=[]
                entry.append(i)
                entry.append(address)
                relepairs.append(entry)

#get news article text/title from url
title,news=getArticleNYT(url)

def featureExtractor(relepairs):