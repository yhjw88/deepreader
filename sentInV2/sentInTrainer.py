import sys
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

def getCorpus(inFilename):
    corpus = []
    with open(inFilename, 'rb') as inFile:
        data = json.load(inFile)
        print "Found {} articles".format(len(data))
        for i, datum in enumerate(data):
            corpus.append(datum["text"])
    return corpus

# Lsa cosine similarities.
def getLsaMatrix(corpus, numComponents):
    tf = TfidfVectorizer(analyzer='word',
                         ngram_range=(1,3),
                         min_df = 0,
                         stop_words = 'english',
                         decode_error='ignore')
    tfidfMatrix = tf.fit_transform(corpus)
    svd = TruncatedSVD(numFeatures)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    lsaMatrix = lsa.fit_transform(tfidfMatrix)
    cosineSimMatrix = linear_kernel(lsaMatrix, lsaMatrix)
    return cosineSimMatrix

def main():
    corpus = getCorpus("data/wArticlesCleaned/wArticlesCleaned.0.json")
    getLsaMatrix(corpus, 100)



if __name__ == "__main__":
    main()
