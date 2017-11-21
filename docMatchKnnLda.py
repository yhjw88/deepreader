import math
import numpy as np
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.neighbors

def main():
    trainX = np.loadtxt("data/docMatchTrainX.txt")
    trainY = np.loadtxt("data/docMatchTrainY.txt")
    testX = np.loadtxt("data/docMatchTestX.txt")
    testY = np.loadtxt("data/docMatchTestY.txt")

    # knn.
    classifier = sklearn.neighbors.KNeighborsClassifier(math.ceil(math.sqrt(len(trainY))))
    classifier.fit(trainX, trainY)
    print "-KNN-"
    print metrics.accuracy_score(testY, classifier.predict(testX))

    #lda
    lda = LinearDiscriminantAnalysis()
    lda.fit(trainX, trainY)
    print "-LDA-"
    print metrics.accuracy_score(testY, lda.predict(testX))

if __name__ == '__main__':
    main()
