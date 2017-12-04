from pymongo import MongoClient
import json
import nltk.data
import random
import re
import string
import sys

def getSections(collection,
                articlesToSkipRange,
                sectionsPerArticleRange,
                minSentencesPerSection,
                numSectionsToUse):
    sections = []
    numToSkip = random.randint(articlesToSkipRange[0], articlesToSkipRange[1])
    for articleDict in collection.find():
        # Skip this article if required.
        if numToSkip != 0:
            numToSkip -= 1
            continue
        else:
            numToSkip = random.randint(articlesToSkipRange[0], articlesToSkipRange[1])
        article = articleDict["text"]

        # Get rid of see also section.
        seeAlsoStartIndex = string.find(article, "==See also==")
        if seeAlsoStartIndex > 0:
            article = article[:seeAlsoStartIndex]

        # Convert into sections and sentences.
        sectionsPerArticle = random.randint(sectionsPerArticleRange[0], sectionsPerArticleRange[1])
        articleSections = []
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        for section in re.split("==[^\n]*==", article):
            sentences = [
                sentence.strip().encode('utf-8')
                for sentence
                in tokenizer.tokenize(section)
                if sentence.strip()
            ]
            if len(sentences) < minSentencesPerSection:
                continue

            articleSections.append(sentences)
            if len(articleSections) >= sectionsPerArticle:
                sections.extend(articleSections)
                if len(sections) % 100 < len(articleSections):
                    print "Got {} sections.".format(len(sections))
                    sys.stdout.flush()
                break

        # Check if we're done.
        if len(sections) >= numSectionsToUse:
            sections = sections[:numSectionsToUse]
            break

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
        outFile.write(json.dumps(dataList))

def main():
    # Data Format
    # { section: [...], sentences: [...], insertionIndexes: [...] }
    # One insertionIndex per sentence in sentences.
    # insertionIndex is n, 0 <= n <= len(section), or None.
    # 0 section[0] 1 section[1] 2 section[2] 3 section[3] 4
    # Small assumption: If insertionIndex is the same, randomly pick one.
    # All ranges are inclusive.
    numSectionsToUseThousands = 5
    articlesToSkipRange = [0, 10]
    sectionsPerArticleRange = [1, 3]
    minSentencesPerSection = 12
    numCorrectSentencesRange = [1, 2]
    # TODO: numIncorrectSentencesRange = [0, 3]
    outFilePrefix = "data/sentIn"

    client = MongoClient()
    collection = client.cs229.wArticlesCleaned

    # Split data.
    numSectionsToUse = int(numSectionsToUseThousands) * 1000
    numSectionsTest = int(numSectionsToUseThousands * 0.2) * 1000
    numSectionsDev = numSectionsTest
    numSectionsTrain = numSectionsToUse - numSectionsTest - numSectionsDev
    print "Train: {} Dev: {} Test: {}".format(numSectionsTrain, numSectionsDev, numSectionsTest)
    sys.stdout.flush()

    # First get all our required sections.
    prelimSections = getSections(collection,
                                 articlesToSkipRange,
                                 sectionsPerArticleRange,
                                 minSentencesPerSection,
                                 numSectionsToUse)

    # Go through the sections and extract sentences.
    dataList = []
    for prelimSection in prelimSections:
        data = {}
        numCorrectSentences = random.randint(numCorrectSentencesRange[0], numCorrectSentencesRange[1])
        data["sentences"] = random.sample(prelimSection, numCorrectSentences)
        data["section"] = [sentence for sentence in prelimSection if (sentence not in data["sentences"])]
        random.shuffle(data["section"])
        data["insertionIndexes"] = getInsertionIndexes(data["sentences"],  prelimSection)
        dataList.append(data)

    # Shuffle data and output.
    random.shuffle(dataList)
    dumpToFile(dataList[:numSectionsTrain], outFilePrefix + "Train.json")
    dumpToFile(dataList[numSectionsTrain:(numSectionsTrain + numSectionsDev)], outFilePrefix + "Dev.json")
    dumpToFile(dataList[(numSectionsTrain + numSectionsDev):], outFilePrefix + "Test.json")

if __name__ == "__main__":
    main()
