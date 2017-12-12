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
    articles = []
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

        # Extract sections to use.
        if len(articleSections) == 0:
            continue
        sectionsPerArticle = random.randint(sectionsPerArticleRange[0], sectionsPerArticleRange[1])
        sectionsToUseThisArticle = random.sample(articleSections, min(sectionsPerArticle, len(articleSections)))
        sections.extend(sectionsToUseThisArticle)
        articles.extend(len(sectionsToUseThisArticle) * [articleSentences])
        if len(sections) % 100 < len(sectionsToUseThisArticle):
            print "Got {} sections.".format(len(sections))
            sys.stdout.flush()

        # Check if we're done.
        if len(sections) >= numSectionsToUse:
            sections = sections[:numSectionsToUse]
            articles = articles[:numSectionsToUse]
            break

    return sections, articles

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

def main():
    # Data Format
    # { prelimSection: [...] article: [...] section: [...], sentences: [...], insertionIndexes: [...] }
    # One insertionIndex per sentence in sentences.
    # insertionIndex is n, 0 <= n <= len(section), or None.
    # 0 section[0] 1 section[1] 2 section[2] 3 section[3] 4
    # Small assumption: If insertionIndex is the same, randomly pick one.
    # All ranges are inclusive.
    numSectionsTrain = 24000
    numSectionsDev = 8000
    numSectionsTest = 8000
    articlesToSkipRange = [0, 10]
    sectionsPerArticleRange = [1, 1]
    minSentencesPerSection = 10
    numCorrectSentencesRange = [1, 1]
    # TODO: numIncorrectSentencesRange = [0, 3]
    outFilePrefix = "data/sentIn"

    client = MongoClient()
    collection = client.cs229.wArticlesCleaned

    print "Train: {} Dev: {} Test: {}".format(numSectionsTrain, numSectionsDev, numSectionsTest)
    sys.stdout.flush()

    # First get all our required sections.
    prelimSections, articles = getSections(collection,
                                           articlesToSkipRange,
                                           sectionsPerArticleRange,
                                           minSentencesPerSection,
                                           numSectionsTrain + numSectionsDev + numSectionsTest)

    # Go through the sections and extract sentences.
    dataList = []
    for prelimSection, article in zip(prelimSections, articles):
        data = {}
        numCorrectSentences = random.randint(numCorrectSentencesRange[0], numCorrectSentencesRange[1])
        data["prelimSection"] = prelimSection
        data["article"] = article
        data["sentences"] = random.sample(prelimSection, numCorrectSentences)
        random.shuffle(data["sentences"])
        data["section"] = [sentence for sentence in prelimSection if (sentence not in data["sentences"])]
        data["insertionIndexes"] = getInsertionIndexes(data["sentences"],  prelimSection)
        dataList.append(data)

    # Shuffle data and output.
    random.shuffle(dataList)
    dumpToFile(dataList[:numSectionsTrain], outFilePrefix + "Train.json")
    dumpToFile(dataList[numSectionsTrain:(numSectionsTrain + numSectionsDev)], outFilePrefix + "Dev.json")
    dumpToFile(dataList[(numSectionsTrain + numSectionsDev):], outFilePrefix + "Test.json")

if __name__ == "__main__":
    main()
