# -*- coding: utf-8 -*-
import pymongo
from pymongo import MongoClient
import re
from sets import Set
import string
import sys

def tagWithDistance1Links(nCollection, wCollection, wCollectionCleaned, numToTag=float('inf')):
    numTagged = 0
    wTitleToIdCache = {}
    for nArticle in nCollection.find().sort([("wikipediaId", pymongo.ASCENDING)]):
        if "wikipediaId1" in nArticle:
            numTagged += 1
            print "Tagged {} article(s) so far, last one skipped, num links: {}".format(numTagged, len(nArticle["wikipediaId1"]))
            sys.stdout.flush()
            continue

        wikipediaId1 = Set()
        for wikipediaId in nArticle["wikipediaId"]:
            # Get the article.
            wArticle = wCollection.find_one({"_id": wikipediaId})

            # Parse all links from the article.
            linkStrings = re.findall("\[\[[^\[\]]*\]\]", wArticle["text"])
            for linkString in linkStrings:
                # Extract the article title.
                barIndex = string.find(linkString, "|")
                if barIndex == -1:
                    barIndex = -2
                wTitle = linkString[2:barIndex]

                # Get the id from title.
                if wTitle not in wTitleToIdCache:
                    wArticle1 = wCollection.find_one({"title": wTitle})
                    wTitleToIdCache[wTitle] = None
                    if wArticle1 and not wArticle1["text"].startswith("#"):
                        wArticle1 = wCollectionCleaned.find_one({"_id": wArticle1["_id"]})
                        wTitleToIdCache[wTitle] = wArticle1["_id"] if wArticle1 else None
                if wTitleToIdCache[wTitle]:
                    wikipediaId1.add(wTitleToIdCache[wTitle])

        # Do the update.
        nCollection.update_one({"_id": nArticle["_id"]}, {"$set": {"wikipediaId1": list(wikipediaId1)}})

        # Print progress.
        numTagged += 1
        print "Tagged {} article(s) so far, last num links: {}".format(numTagged, len(wikipediaId1))
        sys.stdout.flush()
        if numTagged >= numToTag:
            break
    return numTagged

def main():
    client = MongoClient()
    nCollection = client.cs229.nytArticles
    wCollection = client.cs229.wArticles
    wCollectionCleaned = client.cs229.wArticlesCleaned
    tagWithDistance1Links(nCollection, wCollection, wCollectionCleaned, 1000)

if __name__ == "__main__":
    main()
