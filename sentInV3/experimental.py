import json
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import sys
import ucs
import sentInUcs

NUM_FEATURES = 3

# TfIdf cosine similarities.
def getTfIdfCosineSimMatrix(corpus):
    tf = TfidfVectorizer(analyzer='word',
                         ngram_range=(1,3),
                         min_df = 0,
                         stop_words = 'english',
                         decode_error='ignore')
    tfidfMatrix = tf.fit_transform(corpus)
    cosineSimMatrix = linear_kernel(tfidfMatrix, tfidfMatrix)
    return cosineSimMatrix

# Lsa cosine similarities.
def getLsaCosineSimMatrix(corpus, numFeatures):
    tf = TfidfVectorizer(analyzer='word',
                         ngram_range=(1,3),
                         min_df = 0,
                         stop_words = 'english',
                         decode_error='ignore')
    tfidfMatrix = tf.fit_transform(corpus)
    svd = TruncatedSVD(numFeatures)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    lsaMatrix = lsa.fit_transform(tfidfMatrix)
    cosineSimMatrix = linear_kernel(lsaMatrix, lsaMatrix)
    return cosineSimMatrix

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

def makePositive(z):
    # return 1 / (1 + np.exp(-1 * z))
    return np.exp(z)

# Gets feature vectors for actions.
def getFeatures(s1, s2, sI, corpus, cosineSimMatrix):
    features = np.zeros(NUM_FEATURES)
    index1 = corpus.index(s1)
    index2 = corpus.index(s2)
    if sI is None:
        features[0] = 1
        features[1] = 2 * cosineSimMatrix[index1][index2]
        features[2] = features[1]**2
    else:
        indexI = corpus.index(sI)
        features[0] = 1
        features[1] = cosineSimMatrix[index1][indexI] + cosineSimMatrix[indexI][index2]
        features[2] = features[1]**2
    return features

# Sums up feature vectors for actions.
def sumFeatures(sentences, section, corpus, cosineSimMatrix, weights):
    sumFeatures = np.zeros(NUM_FEATURES)
    for index, sentence in enumerate(section[1:-2], 1):
        features = None
        if sentence in sentences:
            features = getFeatures(section[index - 1], section[index + 1], sentence, corpus, cosineSimMatrix)
        else:
            features = getFeatures(section[index - 1], section[index], None, corpus, cosineSimMatrix)
        sumFeatures += np.exp(np.dot(weights, features)) * features
    return sumFeatures

def train():
    # Load data.
    filename = "data/sentInTrain.json"
    dataList = []
    with open(filename, 'rb') as inFile:
        dataList = json.load(inFile)

    with open("data/test2.txt", 'wb') as outFile:
        for sentence in dataList[0]["section"]:
            outFile.write(sentence)
            outFile.write("\n")
        outFile.write("\n")
        for sentence in dataList[0]["prelimSection"]:
            outFile.write(sentence)
            outFile.write("\n")

    # Go through data.
    weights = np.zeros(NUM_FEATURES)
    for i in range(10):
        for dataNum, data in enumerate(dataList):
            corpus = list(data["prelimSection"])
            cosineSimMatrix = getTfIdfCosineSimMatrix(corpus)
            def cost(s1, s2, sI):
                return makePositive(np.dot(weights, getFeatures(s1, s2, sI, data["prelimSection"], cosineSimMatrix)))
            predictedSection = sentInUcs.InsertSentences(list(data["sentences"]), list(data["section"]), cost)

            # Update weights.
            # print "True: \n{}".format(sumFeatures(data["sentences"], data["prelimSection"], getFeatures))
            # print "Predicted: \n{}".format(sumFeatures(data["sentences"], predictedSection, getFeatures))
            oldWeights = weights
            weights -= sumFeatures(data["sentences"], data["prelimSection"], data["prelimSection"], cosineSimMatrix, oldWeights)
            weights += sumFeatures(data["sentences"], predictedSection, data["prelimSection"], cosineSimMatrix, oldWeights)

        # Show progress.
        if i * len(dataList) % 100 == 0:
            print "Ran {} iters".format(i)
            sys.stdout.flush()

    return weights

def getPredicted(data, weights):
    corpus = list(data["prelimSection"])
    cosineSimMatrix = getTfIdfCosineSimMatrix(corpus)
    def cost(s1, s2, sI):
        return makePositive(np.dot(weights, getFeatures(s1, s2, sI, data["prelimSection"], cosineSimMatrix)))
    return sentInUcs.InsertSentences(list(data["sentences"]), list(data["section"]), cost)

def main():
    # Switch for algorithms
    # 0 - ucs with tfidf (Original so slow)
    # 1 - ucs with tfidf
    # 2 - ucs with lsa (needs more data, probably)
    # 3 - Structured Perceptron with tfidf features
    # 4 - ucs with tfidf over whole article
    algoNum = 1

    # Solve utf errors.
    reload(sys)
    sys.setdefaultencoding('utf8')

    # Train if necessary.
    if algoNum == 3:
        weights = train()
        print weights

    # Load data.
    filename = "data/sentInDev.json"
    dataList = []
    with open(filename, 'rb') as inFile:
        dataList = json.load(inFile)

    # Go through data.
    sectionsToTest = len(dataList) * 1.0
    sumSentencesPerSection = 0.0
    numToInsert = 0.0
    numInserted = 0.0
    numInsertedCorrectly = 0.0
    sumSentencesAway = 0.0
    for dataNum, data in enumerate(dataList):
        # Call the algorithm.
        predictedSection = None
        if algoNum == 0:
            ucs.corpus = data["section"] + data["sentences"]
            predictedSection = ucs.InsertSentences(list(data["sentences"]), list(data["section"]), ucs.reward)
        elif algoNum == 1:
            corpus = list(data["prelimSection"])
            cosineSimMatrix = getTfIdfCosineSimMatrix(corpus)
            def cost1(s1, s2, sI):
                index1 = data["prelimSection"].index(s1)
                index2 = data["prelimSection"].index(s2)
                if sI is None:
                    return 1.0 / (2 * (cosineSimMatrix[index1][index2] + 1))
                else:
                    indexI = data["prelimSection"].index(sI)
                    return 1.0 / (cosineSimMatrix[index1][indexI] + 1 + cosineSimMatrix[indexI][index2] + 1)
            predictedSection = sentInUcs.InsertSentences(list(data["sentences"]), list(data["section"]), cost1)
        elif algoNum == 2:
            corpus = list(data["prelimSection"])
            avgNumWords = sum([len(sentence.split()) for sentence in corpus]) / len(corpus)
            cosineSimMatrix = getLsaCosineSimMatrix(corpus, avgNumWords / 2)
            def cost2(s1, s2, sI):
                index1 = data["prelimSection"].index(s1)
                index2 = data["prelimSection"].index(s2)
                if sI is None:
                    return 1.0 / (2 * (cosineSimMatrix[index1][index2] + 1))
                else:
                    indexI = data["prelimSection"].index(sI)
                    return 1.0 / (cosineSimMatrix[index1][indexI] + 1 + cosineSimMatrix[indexI][index2] + 1)
            predictedSection = sentInUcs.InsertSentences(list(data["sentences"]), list(data["section"]), cost2)
        elif algoNum == 3:
            predictedSection = getPredicted(data, weights)
        elif algoNum == 4:
            corpus = list(data["article"])
            cosineSimMatrix = getTfIdfCosineSimMatrix(corpus)
            def cost1(s1, s2, sI):
                index1 = data["article"].index(s1)
                index2 = data["article"].index(s2)
                if sI is None:
                    return 1.0 / (2 * (cosineSimMatrix[index1][index2] + 1))
                else:
                    indexI = data["article"].index(sI)
                    return 1.0 / (cosineSimMatrix[index1][indexI] + 1 + cosineSimMatrix[indexI][index2] + 1)
            predictedSection = sentInUcs.InsertSentences(list(data["sentences"]), list(data["section"]), cost1)
        else:
            print "Algo {} not implemented".format(algoNum)
            sys.exit(1)

        # Update metrics.
        sumSentencesPerSection += len(data["section"])
        actualIndexes = getInsertionIndexes(data["sentences"], predictedSection)
        for expectedIndex, actualIndex in zip(data["insertionIndexes"], actualIndexes):
            numToInsert += 1
            if actualIndex is None:
                continue

            numInserted += 1
            if actualIndex == expectedIndex:
                numInsertedCorrectly += 1
            sumSentencesAway += abs(expectedIndex - actualIndex)

        # Show progress.
        if dataNum % 100 == 0:
            print "Ran {} sections".format(dataNum)
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
        #     for sentence in predictedSection:
        #         outFile.write(sentence)
        #         outFile.write("\n")

    print (
        "sectionsTested: {0:.0f}, avgInsertionPoints: {1:.4f}, sentencesToInsert: {2:.0f}, " +
        "numActuallyInserted: {3:.0f}, avgInsertedCorrectly: {4:.4f}, avgSentencesAway: {5:.4f}"
    ).format(
        sectionsToTest,
        sumSentencesPerSection / sectionsToTest + 1,
        numToInsert,
        numInserted,
        numInsertedCorrectly / numInserted,
        sumSentencesAway / numInserted
    )

if __name__ == "__main__":
    main()
