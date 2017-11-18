from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import json

path = './articles/articles'
ext = '.json'
n_articles = 3
dev_lim = 10000

class TFIDF(object):
    
    def __init__(self):
        pass

    def build_corpus():
        corpus = []

    #build corpus 
    def get_corpus(self):
        corpus = []
        count = 0
        for i in range(n_articles):
            articles = json.load(open(path + str(i) + ext))
            for article in articles:
                if count == dev_lim: return corpus
                corpus.append((dict(article)['title'],dict(article)['text']))
                count += 1
        return corpus

    #build tf-idf
    def get_tfidf(self,corpus):
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english', decode_error='ignore')
        return tf.fit_transform(corpus)

    def relevancy(self,tfidf, indices, sentence1, sentence2):
        index1 = indices.index(sentence1)
        index2 = indices.index(sentence2)
        cos_sim = linear_kernel(tfidf[index1:index1+1], tfidf).flatten()
        return cos_sim[index2]
