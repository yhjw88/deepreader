from pymongo import MongoClient
import nltk.data
import random
import re
import string
import sys
import time
import ucs


def sectionGenerator(collection, sectionsToGenerate, articlesToSkip, minSentencesPerSection):
    sectionNum = 0
    for articleDict in collection.find().skip(articlesToSkip):
        article = articleDict["text"]

        # Get rid of see also section.
        seeAlsoStartIndex = string.find(article, "==See also==")
        if seeAlsoStartIndex > 0:
            article = article[:seeAlsoStartIndex]

        # Convert into sections and sentences.
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        for section in re.split("==[^\n]*==", article):
            if sectionNum >= sectionsToGenerate:
                return

            sentences =  [
                sentence.strip().encode('utf-8') 
                for sentence 
                in tokenizer.tokenize(section) 
                if sentence.strip()
            ]
            if len(sentences) < minSentencesPerSection:
                continue
            else:
                sectionNum += 1
            yield sentences

def getInsertionIndexes(inputSentences, newSection):
    insertionIndexes = len(inputSentences) * [None]
    currentInsertionIndex = 0
    for sentence in newSection:
        if sentence in inputSentences:
            insertionIndexes[inputSentences.index(sentence)] = currentInsertionIndex
        else:
            currentInsertionIndex += 1
    return insertionIndexes

def main():
    sectionsToTest = 100
    articlesToSkip = 10
    numInputSentences = 2
    verbose = 0

    client = MongoClient()
    collection = client.cs229.wArticlesCleaned

    numErrors = 0.0
    sumNormalizedProximity = 0.0
    sumSectionLength = 0.0
    for sectionNum, section in enumerate(sectionGenerator(collection,
                                                          sectionsToTest,
                                                          articlesToSkip,
                                                          numInputSentences * 2)):
        # Set up insertion problem.
        start = time.time()
        expectedSection = section
        inputSentences = random.sample(section, numInputSentences)
        inputPara = [sentence for sentence in section if (sentence not in inputSentences)]
        if verbose:
            print "Set up {} took {} s.".format(sectionNum, time.time() - start)
            start = time.time()
            sys.stdout.flush()

        # Call the algorithm.
        ucs.corpus = inputPara + inputSentences
        actualSection = ucs.InsertSentences(list(inputSentences), inputPara, ucs.reward)
        if verbose:
            print "Test {} took {} s.".format(sectionNum, time.time() - start)
            start = time.time()
            sys.stdout.flush()
        
        # Update results.
        sumSectionLength += len(expectedSection)
        expectedIndexes = getInsertionIndexes(inputSentences, expectedSection)
        actualIndexes = getInsertionIndexes(inputSentences, actualSection)
        for expectedIndex, actualIndex in zip(expectedIndexes, actualIndexes):
            if actualIndex is None:
                numErrors += 1
                sumNormalizedProximity += 1
            else:
                maxProximity = max(expectedIndex, len(expectedSection) - len(inputSentences) - expectedIndex)
                sumNormalizedProximity += abs(expectedIndex - actualIndex) / (1.0 * maxProximity)
                if actualIndex != expectedIndex:
                    numErrors += 1
        if verbose:
            print "Error {} took {} s.".format(sectionNum, time.time() - start)
            start = time.time()
            sys.stdout.flush()

        # Show progress.
        if not verbose and sectionNum % 10 == 0:
            print "Ran {} sections".format(sectionNum)
            sys.stdout.flush()

        # with open("data/test2.txt", 'wb') as outFile:
        #     outFile.write("=============Insertion\n")
        #     for sentence in inputSentences:
        #         outFile.write(sentence)
        #         outFile.write("\n")
        #     outFile.write("=============Expected\n")
        #     for sentence in expectedSection:
        #         outFile.write(sentence)
        #         outFile.write("\n")
        #     outFile.write("=============Actual\n")
        #     for sentence in actualSection:
        #         outFile.write(sentence)
        #         outFile.write("\n")

    avgSectionLength = sumSectionLength / sectionsToTest
    avgError = numErrors / (sectionsToTest * numInputSentences)
    avgNormalizedProximity = sumNormalizedProximity / (sectionsToTest * numInputSentences)
    print "sectionsTested: {}, avgSectionLength: {}, insertionsPerSection: {}, avgError: {}, avgNormalizedProximity: {}".format(
        sectionsToTest,
        avgSectionLength,
        numInputSentences,
        avgError,
        avgNormalizedProximity)

if __name__ == "__main__":
    main()
