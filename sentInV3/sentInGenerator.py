from pymongo import MongoClient
import json
import nltk.data
import random
import re
import sys

# Maximum article id for randomization.
MAXARTICLEID = 55702500
# Minimum article length so we don't pick up stubs.
MINARTICLELENGTH = 10000
# Articles per batch, otherwise too slow.
ARTICLEBATCHSIZE = 100

# Returns:
# list of items, one per section, containing:
# {section: [...], article: [...], articleId: 0}
def getSections(articleCollection,
                articleIdBlacklist,
                minSentencesPerSection,
                numSectionsNeeded):
    articleIdCache= {}
    sections = []
    articleBatch = []
    articleBatchIndex = 0
    while len(sections) < numSectionsNeeded:
        # Get articles.
        articleBatchIndex += 1
        if articleBatchIndex >= len(articleBatch):
            randArticleId = random.randint(0, MAXARTICLEID)
            articleBatch = list(articleCollection.find({"_id": {"$gt": randArticleId}}).sort([("_id", 1)]).limit(ARTICLEBATCHSIZE))
            articleBatchIndex = 0
        articleDict = articleBatch[articleBatchIndex]

        # Check whether article is legal.
        if articleDict["_id"] in articleIdBlacklist or articleDict["_id"] in articleIdCache:
            continue
        article = articleDict["text"]
        if len(article) <= MINARTICLELENGTH:
            continue

        # Convert into sections and sentences.
        articleSentences = []
        articleSections = []
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        for section in re.split("==[^\n]*==", article):
            sentences = [
                sentence.strip().encode('utf-8')
                for sentence
                in tokenizer.tokenize(section)
                if sentence.strip()
            ]
            articleSentences.extend(sentences)
            if len(sentences) >= minSentencesPerSection:
                articleSections.append(sentences)
        if len(articleSections) == 0:
            continue

        # Extract sections to use.
        section = random.choice(articleSections)
        sections.append({"section": section, "article": articleSentences, "articleId": articleDict["_id"]})
        articleIdCache[articleDict["_id"]] = 1

        # Progress update.
        if len(sections) % 100 == 0:
            print "Got {} sections.".format(len(sections))
            sys.stdout.flush()

    return sections

# Finds where the given sentences are inserted into the given section.
def getInsertionIndexes(sentences, section):
    insertionIndexes = len(sentences) * [None]
    currentInsertionIndex = 0
    for sentence in section:
        if sentence in sentences:
            insertionIndexes[sentences.index(sentence)] = currentInsertionIndex
        else:
            currentInsertionIndex += 1
    return insertionIndexes

def dumpToFile(dataList, filename):
    with open(filename, 'wb') as outFile:
        print "Dumping {} to {}".format(len(dataList), filename)
        sys.stdout.flush()
        outFile.write(json.dumps(dataList, indent=4))

# Generates data given parameters, returns ids of articles used, format:
# { section: [...] article: [...] articleId: 0 inputSection: [...], inputSentences: [...], insertionIndexes: [...] }
# One insertionIndex per sentence in sentences.
# insertionIndex is n, 0 <= n <= len(section), or None.
# 0 section[0] 1 section[1] 2 section[2] 3 section[3] 4
# Small assumption: If insertionIndex is the same, randomly pick one.
# All ranges are inclusive.
def generateData(numSectionsTrain,
                 numSectionsDev,
                 numSectionsTest,
                 articleIdBlacklist,
                 minSentencesPerSection,
                 numCorrectSentencesRange,
                 numIncorrectSentencesRange,
                 outFilePrefix):
    client = MongoClient()
    collection = client.cs229.wArticlesCleaned

    print "Train: {} Dev: {} Test: {}".format(numSectionsTrain, numSectionsDev, numSectionsTest)
    sys.stdout.flush()

    # First get all our required sections.
    sections = getSections(collection,
                           articleIdBlacklist,
                           minSentencesPerSection,
                           numSectionsTrain + numSectionsDev + numSectionsTest)
    random.shuffle(sections)

    # Go through the sections and extract sentences.
    articleIdCache = {}
    dataList = []
    for section in sections:
        data = {}
        data["section"] = section["section"]
        data["article"] = section["article"]
        data["articleId"] = section["articleId"]

        data["inputSentences"] = []
        numIncorrectSentences = random.randint(numIncorrectSentencesRange[0], numIncorrectSentencesRange[1])
        while len(data["inputSentences"]) < numIncorrectSentences:
            incorrectSection = random.choice(sections)
            if incorrectSection["articleId"] == section["articleId"]:
                continue
            incorrectSentence = random.choice(incorrectSection["article"])
            if incorrectSentence in data["inputSentences"]:
                continue
            data["inputSentences"].append(incorrectSentence)
        numCorrectSentences = random.randint(numCorrectSentencesRange[0], numCorrectSentencesRange[1])
        data["inputSentences"].extend(random.sample(data["section"], numCorrectSentences))
        random.shuffle(data["inputSentences"])

        data["inputSection"] = [sentence for sentence in data["section"] if (sentence not in data["inputSentences"])]
        data["insertionIndexes"] = getInsertionIndexes(data["inputSentences"],  data["section"])
        dataList.append(data)
        articleIdCache["articleId"] = 1

    # Output.
    dumpToFile(dataList[:numSectionsTrain], outFilePrefix + "Train.json")
    dumpToFile(dataList[numSectionsTrain:(numSectionsTrain + numSectionsDev)], outFilePrefix + "Dev.json")
    dumpToFile(dataList[(numSectionsTrain + numSectionsDev):], outFilePrefix + "Test.json")
    return articleIdCache

def main():
   articleIdCache = generateData(numSectionsTrain = 60000,
                                 numSectionsDev = 20000,
                                 numSectionsTest = 20000,
                                 articleIdBlacklist = {},
                                 minSentencesPerSection = 10,
                                 numCorrectSentencesRange = [1, 2],
                                 numIncorrectSentencesRange = [0, 2],
                                 outFilePrefix = "data/sentIn/sentInD1")

   articleIdCache = generateData(numSectionsTrain = 0,
                                 numSectionsDev = 50000,
                                 numSectionsTest = 50000,
                                 articleIdBlacklist = articleIdCache,
                                 minSentencesPerSection = 10,
                                 numCorrectSentencesRange = [1, 1],
                                 numIncorrectSentencesRange = [0, 0],
                                 outFilePrefix = "data/sentIn/sentInD2")

if __name__ == "__main__":
    main()
