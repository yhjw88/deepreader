import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sentInUcs as sentInUcs
import sentInTester

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

def testWith(tfIdfArticle, alpha, filename):
    # Load data.
    dataList = []
    with open(filename, 'rb') as inFile:
        dataList = json.load(inFile)

    # Get the cost data given the data.
    def extractCostData(data):
        corpus = []
        if tfIdfArticle:
            corpus.extend(data["article"])
            for sentence in data["inputSentences"]:
                if sentence not in data["section"]:
                    corpus.append(sentence)
        else:
            corpus.extend(data["inputSection"])
            corpus.extend(data["inputSentences"])
        costData = {
            "corpus": corpus,
            "cosineSimMatrix": getTfIdfCosineSimMatrix(corpus)
        }
        return costData
    
    # Get the cost given the costData.
    def cost(s1, s2, sI, costData):
        corpus = costData["corpus"]
        cosineSimMatrix = costData["cosineSimMatrix"]
        index1 = corpus.index(s1)
        index2 = corpus.index(s2)
        if sI is None:
            return 1.0 / (alpha * cosineSimMatrix[index1][index2] + 1)
        else:
            indexI = corpus.index(sI)
            return 1.0 / (cosineSimMatrix[index1][indexI] + cosineSimMatrix[indexI][index2] + 1)

    # Test on data
    stats = sentInTester.testOn(sentInUcs.InsertSentences, dataList, cost, extractCostData)
    return stats

def plot(lgAlphas, precisions, recalls, f1s, filename):
    plt.figure()
    plt.plot(lgAlphas, precisions, "g+", label="precision")
    plt.plot(lgAlphas, recalls, "b+", label="recall")
    plt.plot(lgAlphas, f1s, "k+", label="f1")
    plt.xlabel("lgAlpha")
    plt.ylabel("fraction")
    plt.legend()
    plt.savefig(filename)
    return

def plot2(lgAlphas, avgSentencesAway, filename):
    plt.figure()
    plt.plot(lgAlphas, avgSentencesAway, "k+")
    plt.xlabel("lgAlpha")
    plt.ylabel("avgSentencesAway")
    plt.savefig(filename)
    return

def main():
    tfIdfArticle = True
    plotAlpha = False
    dataFilename = "data/sentIn/sentInD1Test.json"
    plotFilename = "data/sentIn/sentInD1PlotF1.png"
    plot2Filename = "data/sentIn/sentInD1PlotR.png"

    if not plotAlpha:
        testWith(tfIdfArticle, 0.25, dataFilename)
        return
    
    lgAlphas = range(-8, 23, 2)
    precisions = []
    recalls = []
    f1s = []
    avgSentencesAway = []
    for lgAlpha in lgAlphas:
        stats = testWith(tfIdfArticle, 2**lgAlpha, dataFilename)
        precisions.append(stats["precision"])
        recalls.append(stats["recall"])
        f1s.append(stats["f1"])
        avgSentencesAway.append(stats["avgSentencesAway"])
    plot(lgAlphas, precisions, recalls, f1s, plotFilename)
    plot2(lgAlphas, avgSentencesAway, plot2Filename)

if __name__ == "__main__":
    main()
