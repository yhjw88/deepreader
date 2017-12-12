# -*- coding: utf-8 -*-
import json
import pymongo
from pymongo import MongoClient
import re
import string
import sys
import WikiExtractor

def cleanArticle(inCollection, outCollection):
    lastId = -1
    if outCollection.count() != 0:
        lastId = outCollection.find().sort([("_id", pymongo.DESCENDING)]).limit(1)[0]["_id"]
        print "Starting from id greater than: {}".format(lastId)
        sys.stdout.flush()
    numCleaned = 0
    for article in inCollection.find({"_id": {"$gt": lastId}}).sort([("_id", pymongo.ASCENDING)]):
        # Parse it.
        extractor = WikiExtractor.Extractor(article["_id"], article["title"], [article["text"]])
        article["text"] = extractor.clean()
        outCollection.insert_one(article)

        # Print progress.
        numCleaned += 1
        if numCleaned % 1000 == 0:
            print "Cleaned {} articles so far...".format(numCleaned)
            sys.stdout.flush()

    return numCleaned

# Given a reference itemString, will parse out the value of the given name if present.
# For example parseFromReferenceItem("a=akldjfa", "a") will return "akldjfa".
# Returns None if not present.
def parseFromReferenceItem(itemString, name):
    # Quick return if the name is not correct.
    if string.find(itemString, name) < 0:
        return None

    # Find the start of the value.
    valueStartIndex = string.find(itemString, name) + len(name)
    valueStartIndex = string.find(itemString, "=", valueStartIndex) + 1

    # Make sure the name is correct.
    actualName = itemString[:valueStartIndex - 1]
    if name != string.strip(actualName):
        return None

    # Parse out the value if the name is correct.
    value = itemString[valueStartIndex:]
    return string.strip(value)

def extractReferencesFromArticles(inCollection, outCollection, query={}):
    numExtracted = 0
    for article in inCollection.find(query).sort([("_id", pymongo.ASCENDING)]):
        # Start a reference list.
        references = []

        # Fetch all the references.
        referenceStrings = re.findall("{{cite[^{}]*}}", article["text"])
        for referenceString in referenceStrings:
            # Things we're interested in.
            reference = {"raw": referenceString}

            # Try to extract values we're interested in.
            currentStartIndex = 0
            while True:
                # Get the next item.
                nextBar = string.find(referenceString, "|", currentStartIndex)
                currentEndIndex = nextBar if nextBar != -1 else -2
                itemString = referenceString[currentStartIndex:currentEndIndex]

                # See if has values we're interested in.
                url = parseFromReferenceItem(itemString, "url")
                if url:
                    reference["url"] = url
                title = parseFromReferenceItem(itemString, "title")
                if title:
                    reference["title"] = title

                # Exit the loop if there are no more items, continue if there are.
                if nextBar == -1:
                    break
                currentStartIndex = nextBar + 1

            # Add to our reference list.
            references.append(reference)

        # Do the update.
        outCollection.update_one({"_id": article["_id"]}, {"$set": {"references": references}})

        # Stats.
        numExtracted += 1
        if numExtracted % 1000 == 0:
            print "Extracted references from {} articles so far, lastId: {} ...".format(numExtracted, article["_id"])
            sys.stdout.flush()

    return numExtracted

def dumpArticlesFromDatabase(collection, filePrefix, query={}, numArticles=float('inf'), articlesPerFile=100000):
    # Helper function to dump a list of articles, and then clear the list.
    def dumpArticleList(articleList, fileNum):
        filename = filePrefix + "." + str(fileNum) + ".json"
        print "Dumping {} articles to {}, lastId: {} ...".format(len(articleList), filename, articleList[-1]["_id"])
        sys.stdout.flush()
        with open(filename, 'wb') as outFile:
            outFile.write(json.dumps(articleList))
        del articleList[:]

    # Main loop to iterate through all articles.
    articleList = []
    fileNum = 0
    for articleNum, article in enumerate(collection.find(query).sort([("_id", pymongo.ASCENDING)])):
        if articleNum % articlesPerFile == 0 and articleNum != 0:
            dumpArticleList(articleList, fileNum)
            fileNum += 1
        articleList.append(article)
        if articleNum >= numArticles - 1:
            break
    if len(articleList) > 0:
        dumpArticleList(articleList, fileNum)

    # Hacky due to python scoping.
    return articleNum + 1

def main():
    client = MongoClient()
    inCollection = client.cs229.wArticles
    # outCollection = client.cs229.wArticlesCleaned
    outCollection = client.cs229.nytArticles
    cleanArticles = False
    extractReferences = False
    testArticle = False
    dumpDatabase = True
    examineDump = False

    if cleanArticles:
        numCleaned = cleanArticle(inCollection, outCollection)
        print "Total articles cleaned: {}".format(numCleaned)

    if extractReferences:
        numExtracted = extractReferencesFromArticles(inCollection, outCollection, {"_id": {"$gt": -1}})
        print "Total articles with references extracted: {}".format(numExtracted)

    if testArticle:
        with open("test.txt", 'wb') as outFile:
            article = outCollection.find_one({"text": {"$regex": ".*References.*"}})
            outFile.write(article["text"].encode('utf8'))

    if dumpDatabase:
        # numDumped = dumpArticlesFromDatabase(outCollection, "data/wArticlesCleaned/wArticlesCleaned", {"_id": {"$gt": -1}})
        numDumped = dumpArticlesFromDatabase(outCollection, "data/nytArticles")
        print "Total articles dumped: {}".format(numDumped)

    if examineDump:
        filename = "data/wArticlesCleaned/wArticlesCleaned.41.json"
        with open(filename, 'rb') as inFile:
            data = json.load(inFile)
            print len(data)
            for i, datum in enumerate(data):
                if datum["references"]:
                    print datum["references"]
                if i == 5:
                    break

if __name__ == "__main__":
    main()
