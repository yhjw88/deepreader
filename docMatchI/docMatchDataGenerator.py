from __future__ import division
import collections
import json
import pymongo
from pymongo import MongoClient
from random import shuffle
import sys

# Creates 4 wrongs for every right.
def createWNPairs(nArticles, wArticles):
    wnPairs = []
    nIdSet = collections.defaultdict(int)
    for nArticle in nArticles:
        shuffle(wArticles)
        for wArticle in wArticles:
            if wArticle["_id"] in nArticle["wikipediaId"]:
                wnPairs.append([wArticle["_id"], nArticle["url"], 1])
            elif nIdSet[nArticle["url"]] <= 4:
                wnPairs.append([wArticle["_id"], nArticle["url"], -1])
                nIdSet[nArticle["url"]] += 1
    return wnPairs

def dumpToFile(dataList, filename):
    with open(filename, 'wb') as outFile:
        outFile.write(json.dumps(dataList, indent=4))
    print "Finished dumping {} data to file".format(len(dataList))
    sys.stdout.flush()

def main():
    client = MongoClient()
    wCollection = client.cs229.wArticlesCleaned
    nCollection = client.cs229.nytArticles

    # Get references.
    nArticles = list(nCollection.find().sort([("wikipediaId", pymongo.ASCENDING)]).limit(10000))

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
    print len(nArticles)
    print len(wArticles)
    print "Finished fetching data"
    sys.stdout.flush()

    # Split into train, dev, and test.
    shuffle(nArticles)
    nArticlesTrain = nArticles[:6000]
    nArticlesDev = nArticles[6000:8000]
    nArticlesTest = nArticles[8000:]

    # Create the data pairings.
    wnPairsTrain = createWNPairs(nArticlesTrain, wArticles)
    wnPairsDev = createWNPairs(nArticlesDev, wArticles)
    wnPairsTest = createWNPairs(nArticlesTest, wArticles)
    print "Finished creating pairs"
    sys.stdout.flush()

    # Save to file.
    dumpToFile(wnPairsTrain, "data/docMatchITrain.json")
    dumpToFile(wnPairsDev, "data/docMatchIDev.json")
    dumpToFile(wnPairsTest, "data/docMatchITest.json")

if __name__ == '__main__':
    main()
