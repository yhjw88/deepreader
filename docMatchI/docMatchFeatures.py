from __future__ import division
import collections
import numpy as np
import pymongo
from pymongo import MongoClient
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sys

# Creates 4 wrongs for every right.
def createWNPairs(nArticles, wArticles):
    wnPairs = []
    nIdSet = collections.defaultdict(int)
    for nArticle in nArticles:
        shuffle(wArticles)
        for wArticle in wArticles:
            if wArticle["_id"] in nArticle["wikipediaId"]:
                wnPairs.append((wArticle, nArticle))
            elif nIdSet[nArticle["url"]] <= 4:
                wnPairs.append((wArticle, nArticle))
                nIdSet[nArticle["url"]] += 1
    return wnPairs

# Calculate cosine similarities for the given data.
def calculateCosineSimilarities(articles, tf, tfidfMatrixOrigin):
    corpus = []
    for article in articles:
        corpus.append(article["scrapedTitle"])
        article["titleCorpusIndex"] = len(corpus) - 1
        corpus.append(article["scrapedText"])
        article["textCorpusIndex"] = len(corpus) - 1
    tfidfMatrix = tf.transform(corpus)
    return linear_kernel(tfidfMatrix, tfidfMatrixOrigin)

# Extracts cosine similarity features for X, and labels Y.
def extractFeatures(wnPairs, cosineSimMatrix):
    Y = np.zeros(len(wnPairs))
    X = np.zeros((len(wnPairs), 2))
    for i, (wArticle, nArticle) in enumerate(wnPairs):
        Y[i] = 1 if wArticle["_id"] in nArticle["wikipediaId"] else -1
        X[i][0] = cosineSimMatrix[nArticle["titleCorpusIndex"]][wArticle["titleCorpusIndex"]]
        X[i][1] = cosineSimMatrix[nArticle["textCorpusIndex"]][wArticle["textCorpusIndex"]]
    return X, Y

def main():
    client = MongoClient()
    wCollection = client.cs229.wArticlesCleaned
    nCollection = client.cs229.nytArticles

    # Get references.
    nArticles = list(nCollection.find().sort([("wikipediaId", pymongo.ASCENDING)]).limit(5000))

    # Fetch all the linked articles.
    wArticles = []
    wIdSet = {}
    for nArticle in nArticles:
        # Delete the id.
        del nArticle["_id"]

        # Fetch the wikipedia article(s) if necessary.
        for wikipediaId in nArticle["wikipediaId"]:
            if wikipediaId in wIdSet:
                continue
            wIdSet[wikipediaId] = 1
            wArticles.append(wCollection.find_one({"_id": wikipediaId}))
    print "Finished fetching data, nArticles: {}, wArticles: {}".format(len(nArticles), len(wArticles))
    sys.stdout.flush()

    # Split into train, dev, and test.
    shuffle(nArticles)
    nArticlesTrain = nArticles[:3000]
    nArticlesDev = nArticles[3000:4000]
    nArticlesTest = nArticles[4000:]

    # Set up tfidf matrix on training data only.
    corpus = []
    for article in wArticles:
        corpus.append(article["title"])
        article["titleCorpusIndex"] = len(corpus) - 1
        corpus.append(article["text"])
        article["textCorpusIndex"] = len(corpus) - 1
    for article in nArticlesTrain:
        corpus.append(article["scrapedTitle"])
        article["titleCorpusIndex"] = len(corpus) - 1
        corpus.append(article["scrapedText"])
        article["textCorpusIndex"] = len(corpus) - 1
    tf = TfidfVectorizer(analyzer='word',
                         ngram_range=(1,3),
                         min_df = 0,
                         stop_words = 'english',
                         decode_error='ignore')
    tfidfMatrix = tf.fit_transform(corpus)
    cosineSimMatrix = linear_kernel(tfidfMatrix, tfidfMatrix)
    print "Finished tfidf for train"
    sys.stdout.flush()

    # Calculate cosine similarities for dev.
    cosineSimMatrixDev = calculateCosineSimilarities(nArticlesDev, tf, tfidfMatrix)
    print "Finished tfidf for dev"
    sys.stdout.flush()

    # Calculate cosine similarities for test.
    cosineSimMatrixTest = calculateCosineSimilarities(nArticlesTest, tf, tfidfMatrix)
    print "Finished tfidf for test"
    sys.stdout.flush()

    # Create the (w, n) pairs.
    wnPairsTrain = createWNPairs(nArticlesTrain, wArticles)
    wnPairsDev = createWNPairs(nArticlesDev, wArticles)
    wnPairsTest = createWNPairs(nArticlesTest, wArticles)
    print "Finished creating pairs"
    sys.stdout.flush()

    # Extract features for training data.
    XTrain, YTrain = extractFeatures(wnPairsTrain, cosineSimMatrix)
    np.savetxt("data/docMatchITrainX.txt", XTrain)
    np.savetxt("data/docMatchITrainY.txt", YTrain)
    print "Outputted training data, {}".format(len(YTrain))
    sys.stdout.flush()

    # Extract features for dev data.
    XDev, YDev = extractFeatures(wnPairsDev, cosineSimMatrixDev)
    np.savetxt("data/docMatchIDevX.txt", XDev)
    np.savetxt("data/docMatchIDevY.txt", YDev)
    print "Outputted dev data, {}".format(len(YDev))
    sys.stdout.flush()

    # Extract features for test data.
    XTest, YTest = extractFeatures(wnPairsTest, cosineSimMatrixTest)
    np.savetxt("data/docMatchITestX.txt", XTest)
    np.savetxt("data/docMatchITestY.txt", YTest)
    print "Outputted test data, {}".format(len(YTest))
    sys.stdout.flush()

if __name__ == '__main__':
    main()
