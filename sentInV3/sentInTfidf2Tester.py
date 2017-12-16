import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sentInUcs3 as sentInUcs
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

def main():
    tfIdfArticle = True
    filterShort = True
    dataFilename = "data/sentIn/sentInD2Test.json"

    # Load data.
    dataList = []
    with open(dataFilename, 'rb') as inFile:
        dataList = json.load(inFile)

    # Filter short sentences.
    if filterShort:
        tempDataList = []
        for data in dataList:
            shouldFilter = False
            for sentence in data["inputSentences"]:
                if len(sentence) <= 20:
                    shouldFilter = True
            if not shouldFilter:
                tempDataList.append(data)
        dataList = tempDataList

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
        return 1.0 / (cosineSimMatrix[index1][index2] + 1)

    # Test on data
    stats = sentInTester.testOn(sentInUcs.InsertSentences, dataList, cost, extractCostData)
    return stats

if __name__ == "__main__":
    main()
