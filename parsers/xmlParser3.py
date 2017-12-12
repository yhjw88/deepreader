# -*- coding: utf-8 -*-
import pymongo
from pymongo import MongoClient
import re
import sys
from xml.etree.ElementTree import iterparse

# States
NOPAGE = 0
INPAGE = 1
INREVISION = 2

# Extracts the actual tag from elem.tag.
def extractTag(tag):
    return tag[len("{http://www.mediawiki.org/xml/export-0.10/}"):]

# Counts the number of tags with tagName.
def countTags(filename, tagName):
    numTags = 0
    doc = iterparse(filename, ('start', 'end'))
    for event, elem in doc:
        if event == 'start' and extractTag(elem.tag) == tagName:
            numTags  += 1
    return numTags

# Extract the final revision of articles.
def extractArticles(filename, collection, articlesNeeded=float('inf')):
    # Initialize variables.
    currentState = NOPAGE
    articleDict = {}
    skipArticle = False
    stats = {"numStored":0, "numSkipped": 0}
    lastId = collection.find().sort([("_id", pymongo.DESCENDING)]).limit(1)[0]["_id"]

    # Loop through every tag in the document.
    doc = iter(iterparse(filename, ('start', 'end')))
    _, root = doc.next()
    for event, elem in doc:
        if event == 'start':
            extractedTag = extractTag(elem.tag)

            # Tags informing state.
            if currentState == NOPAGE and extractedTag == "page":
                currentState += 1
            if currentState == INPAGE and extractedTag == "revision":
                currentState += 1

        elif event == 'end':
            # Parse XML end events
            extractedTag = extractTag(elem.tag)

            # Tags informing state.
            if extractedTag == "page":
                # Update stats and potentially save article.
                if skipArticle or "text" not in articleDict or "title" not in articleDict:
                    stats["numSkipped"] += 1
                else:
                    collection.insert_one(articleDict)
                    stats["numStored"] += 1
                    if stats["numStored"] >= articlesNeeded:
                        return stats

                # Report progress.
                if not skipArticle and stats["numStored"] % 1000 == 0:
                    print "Stored {} articles so far...".format(stats["numStored"])
                    sys.stdout.flush()
                elif skipArticle and stats["numSkipped"] % 1000 == 0:
                    print "Skipped {} articles so far...".format(stats["numSkipped"])
                    sys.stdout.flush()

                # Page ended, reset state.
                currentState = NOPAGE
                articleDict.clear()
                skipArticle = False

                # Clean memory.
                root.clear()

            elif extractedTag == "revision":
                currentState -= 1

            # Skip further processing if skipping article.
            if skipArticle:
                continue

            # Tags producing information.
            if extractedTag == "title":
                articleDict["title"] = elem.text
                if not articleDict["title"] or re.match("[^ ]*:", articleDict["title"]):
                    skipArticle = True
            elif currentState == INPAGE and extractedTag == "id":
                articleDict["_id"] = long(elem.text)
                if articleDict["_id"] <= lastId:
                    skipArticle = True
            elif extractedTag == "timestamp":
                articleDict["timestamp"] = elem.text
            elif extractedTag == "text":
                articleDict["text"] = elem.text
                if not articleDict["text"] or articleDict["text"].startswith("#REDIRECT"):
                    skipArticle = True

    # Ran out of articles, return some stats
    return stats

def main():
    filename = "data/enwiki-latest-pages-articles.xml"
    countArticles = False
    storeArticles = True

    if countArticles:
        numTags = countTags(filename, "page")
        print "Total articles found: {}".format(numTags)

    if storeArticles:
        client = MongoClient()
        collection = client.cs229.wArticles
        stats = extractArticles(filename, collection)
        print "Total articles stored: {}".format(stats["numStored"])
        print "Total articles skipped: {}".format(stats["numSkipped"])

if __name__ == "__main__":
    main()
