import requests
from bs4 import BeautifulSoup

#def getArticle(url):
url = 'http://money.cnn.com/2017/09/19/technology/gadgets/iphone-8-review/?iid=EL'
web = requests.get(url)
soup=BeautifulSoup(web.text,'lxml')
news = soup.find("div",{"id":"storytext"})
print news.text
#return news

from nltk.stem.porter import *
from nltk.corpus import stopwords
import string

stop = stopwords.words("english")
ps = PorterStemmer()
morestop = {"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "I", "need", "like", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "work", "use", "project", "make", "add", "look", "the", "want", "user", "looking"}

#def preprocess(text):
vocab = set()
for word in news.text.split():
    word = word.lower()
    if word in stop or word in morestop:
        continue
    else:
        w = ''
        for char in word:
            if char not in string.punctuation:
                w += char
        vocab.add(w)
vocab = list(vocab)
#return vocab
print(len(vocab))

import lda
import numpy as np

#def extract_topics(num_topics):
mat = np.zeros((1, len(vocab)), dtype=np.int)
for w in range(len(vocab)):
    for word in news.text.split():
        word = word.lower()
        if word in stop or word in morestop:
            continue
        else:
            w = ''
            for char in word:
                if char not in string.punctuation:
                    w += char
            mat[0][vocab.index(w)] += 1
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(mat)
topic_word = model.topic_word_
n_top_words = 3
num_topics = 10
topics = []
for i, topic_dist in enumerate(topic_word):
    if len(topics) > num_topics-1: break
    sub = []
    for elem in np.argsort(topic_dist).astype(int):
        sub.append(vocab[elem])
    topics.append(sub[:-n_top_words+1:-1])
print(topics)
#return topics

import json
dev_lim = 10000
while True:
    #def load_wikipedia():
    path = './articles/wArticlesCleaned.'
    ext = '.json'
    n_articles = 62
    dev_lim += 10000
    
    wikipedia = []
    count = 0
    for i in range(n_articles):
        articles = json.load(open(path + str(i) + ext))
        for article in articles:
            if count == dev_lim: break
            wikipedia.append((dict(article)['title'],dict(article)['text']))
            count += 1
    print(wikipedia[3][0])
    #return wikipedia

    import copy 

    #def get_corpus():
    corpus = [news.text]
    titles = [soup.title]
    print len(corpus)

    for article in wikipedia:
        # if article[0] in topics:
        corpus.append(article[1])
        titles.append(article[0])
    print len(corpus)
    #return corpus, titles

    from tfidf import TFIDF

    #def get_sim_docs():
    tfi = TFIDF()
    tfidf = tfi.get_tfidf(corpus)

    sim_docs = []
    for index, score in tfi.similar_docs(tfidf, 0, 5):
        sim_docs.append((index, score))
        print score, titles[index]

    print "Most relevant document is " + titles[sim_docs[0][0]]
    #return sim_docs

