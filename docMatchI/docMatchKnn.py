import math
import numpy as np
from sklearn import metrics
import sklearn.neighbors
import sys

def main():
    trainX = np.loadtxt("data/docMatchITrainX.txt")
    trainY = np.loadtxt("data/docMatchITrainY.txt")
    devX = np.loadtxt("data/docMatchIDevX.txt")
    devY = np.loadtxt("data/docMatchIDevY.txt")
    testX = np.loadtxt("data/docMatchITestX.txt")
    testY = np.loadtxt("data/docMatchITestY.txt")

    # Train on dev.
    k = int(math.sqrt(len(trainY)))
    worse = False
    acc1 = 0
    classifier = None
    while not worse:
        classifier = sklearn.neighbors.KNeighborsClassifier(k)
        classifier.fit(trainX, trainY)
        acc2 = metrics.accuracy_score(devY, classifier.predict(devX))
        print "k: {}, dev acc: {}".format(k, acc2)
        sys.stdout.flush()
        if acc2 > acc1:
            k += 100
            acc1 = acc2
        else:
            worse = True
    k -= 100

    # Output on test.
    classifier.fit(testX, testY)
    acc = metrics.accuracy_score(testY, classifier.predict(testX))
    print "final k: {}, test acc: {}".format(k, acc)

if __name__ == '__main__':
    main()
