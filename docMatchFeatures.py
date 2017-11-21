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

# Calculate cosinge similarities for the given data.
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

# Extracts cosine similarity features for X, and labels Y.
def extractFeatures2(wnPairs, cosineSimMatrix):
    Y = np.zeros(len(wnPairs))
    X = np.zeros((len(wnPairs), 4))
    for i, (wArticle, nArticle) in enumerate(wnPairs):
        Y[i] = 1 if wArticle["_id"] in nArticle["wikipediaId"] else -1
        X[i][0] = cosineSimMatrix[nArticle["titleCorpusIndex"]][wArticle["titleCorpusIndex"]]
        X[i][1] = cosineSimMatrix[nArticle["textCorpusIndex"]][wArticle["textCorpusIndex"]]
        X[i][2] = (cosineSimMatrix[nArticle["titleCorpusIndex"]][wArticle["titleCorpusIndex"]])**2
        X[i][3] = (cosineSimMatrix[nArticle["textCorpusIndex"]][wArticle["textCorpusIndex"]])**2
    return X, Y

def main():
    client = MongoClient()
    wCollection = client.cs229.wArticlesCleaned
    nCollection = client.cs229.nytArticles

    # 2000 references for train, 1000 references for test.
    nArticlesTrain = list(nCollection.find().sort([("wikipediaId", pymongo.ASCENDING)]).limit(2400))
    nArticlesDev = list(nCollection.find().sort([("wikipediaId", pymongo.ASCENDING)]).skip(2400).limit(800))
    nArticlesTest = list(nCollection.find().sort([("wikipediaId", pymongo.ASCENDING)]).skip(3200).limit(800))

    # Fetch all the linked articles.
    wArticles = []
    wIdSet = {}
    for nArticle in (nArticlesTrain + nArticlesDev + nArticlesTest):
        # Fetch the wikipedia article(s) if necessary.
        for wikipediaId in nArticle["wikipediaId"]:
            if wikipediaId in wIdSet:
                continue
            wIdSet[wikipediaId] = 1
            wArticles.append(wCollection.find_one({"_id": wikipediaId}))
    print "Finished fetching data"
    sys.stdout.flush()

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

    # Calculate cosine similarities for dev.
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
    np.savetxt("data/docMatchTrainX.txt", XTrain)
    np.savetxt("data/docMatchTrainY.txt", YTrain)
    print "Outputted training data, {}".format(len(YTrain))
    sys.stdout.flush()

    # Extract features for dev data.
    XDev, YDev = extractFeatures(wnPairsDev, cosineSimMatrixDev)
    np.savetxt("data/docMatchDevX.txt", XDev)
    np.savetxt("data/docMatchDevY.txt", YDev)
    print "Outputted dev data, {}".format(len(YDev))
    sys.stdout.flush()

    # Extract features for test data.
    XTest, YTest = extractFeatures(wnPairsTest, cosineSimMatrixTest)
    np.savetxt("data/docMatchTestX.txt", XTest)
    np.savetxt("data/docMatchTestY.txt", YTest)
    print "Outputted test data, {}".format(len(YTest))
    sys.stdout.flush()

    # Extract features 2 for training data.
    XTrain, _ = extractFeatures2(wnPairsTrain, cosineSimMatrix)
    np.savetxt("data/docMatchTrain2X.txt", XTrain)
    print "Outputted training data 2, {}".format(len(YTrain))
    sys.stdout.flush()

    # Extract features 2 for dev data.
    XDev, _ = extractFeatures2(wnPairsDev, cosineSimMatrixDev)
    np.savetxt("data/docMatchDev2X.txt", XDev)
    print "Outputted dev data 2, {}".format(len(YDev))
    sys.stdout.flush()

    # Extract features 2 for test data.
    XTest, _ = extractFeatures2(wnPairsTest, cosineSimMatrixTest)
    np.savetxt("data/docMatchTest2X.txt", XTest)
    print "Outputted test data 2, {}".format(len(YTest))
    sys.stdout.flush()

if __name__ == '__main__':
    main()
