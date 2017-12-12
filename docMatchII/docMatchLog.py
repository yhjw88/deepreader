from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import sys

# Logistic Regression from class.
# Don't reinvent the wheel.
class LogisticRegression:
    def __init__(self, X, Y):
        self.resetData(X, Y)

    def resetData(self, X, Y):
        self.X = self.addIntercept(X)
        self.Y = Y

    def addIntercept(self, X_):
        m, n = X_.shape
        X = np.zeros((m, n + 1))
        X[:, 0] = 1
        X[:, 1:] = X_
        return X

    # Average Logistic Loss.
    def calcGrad(self, X, Y, theta):
        m, n = X.shape
        grad = np.zeros(theta.shape)

        margins = Y * X.dot(theta)
        probs = 1. / (1 + np.exp(margins))
        grad = -(1./m) * (X.T.dot(probs * Y))

        return grad

    def doWork(self, learningRate):
        X = self.X
        Y = self.Y

        m, n = X.shape
        theta = np.zeros(n)

        i = 0
        while True:
            i += 1
            prev_theta = theta
            grad = self.calcGrad(X, Y, theta)
            theta = theta  - learningRate * (grad)
            # theta = theta  - learningRate * (grad + 0.00001 * theta)
            if i % 10000 == 0:
                print 'Iterations: {}, Grad: {}, Theta: {}'.format(i, np.linalg.norm(grad, 1), theta)
                sys.stdout.flush()
            if np.linalg.norm(prev_theta - theta) < 1e-15:
                print('Converged in %d iterations' % i)
                break
        self.theta = theta
        return theta

    def getPredicted(self, testX):
        testX = self.addIntercept(testX)
        scores = np.dot(testX, self.theta)
        return [1 if score >= 0 else -1 for score in scores]

    def plotLin(self, filename):
        X = self.X
        Y = self.Y
        theta = self.theta
        plt.figure()

        # Split the list of x's.
        x1Pluses = []
        x2Pluses = []
        x1Minuses = []
        x2Minuses = []
        for xRow, y in zip(X, Y):
            if y > 0:
                x1Pluses.append(xRow[1])
                x2Pluses.append(xRow[2])
            else:
                x1Minuses.append(xRow[1])
                x2Minuses.append(xRow[2])

        # Get some points for the line.
        linep1x1 = min(X[:,1])
        linep1x2 = -1 * (theta[0] + theta[1] * linep1x1) / theta[2]
        linep2x1 = max(X[:,1])
        linep2x2 = -1 * (theta[0] + theta[1] * linep2x1) / theta[2]

        # Now plot the things.
        plt.plot(x1Pluses, x2Pluses, "b+")
        plt.plot(x1Minuses, x2Minuses, "r.")
        plt.plot([linep1x1, linep2x1], [linep1x2, linep2x2], "g-")
        plt.xlabel("title")
        plt.ylabel("text")
        plt.savefig(filename)
        return

def countCorrect(actualYs, predictedYs):
    count = 0
    for actY, preY in zip(actualYs, predictedYs):
        if actY == preY:
            count += 1
    return count

def reg1():
    # Do regression.
    logisticRegression = LogisticRegression(np.loadtxt("data/docMatchITrainX.txt"), np.loadtxt("data/docMatchITrainY.txt"))
    logisticRegression.doWork(10)
    logisticRegression.plotLin("data/docMatchITrain.png")

    # Dev accuracy
    devX = np.loadtxt("data/docMatchIDevX.txt")
    devY = np.loadtxt("data/docMatchIDevY.txt")
    predictedY = logisticRegression.getPredicted(devX)
    print "Dev correct: {} total: {}".format(countCorrect(devY, predictedY), len(devY))
    logisticRegression.resetData(devX, devY)
    logisticRegression.plotLin("data/docMatchIDev.png")

    # Test accuracy
    # testX = np.loadtxt("data/docMatchTestX.txt")
    # testY = np.loadtxt("data/docMatchTestY.txt")
    # predictedY = logisticRegression.getPredicted(testX)
    # print "Test correct: {} total: {}".format(countCorrect(testY, predictedY), len(testY))

def main():
    reg1()

if __name__ == '__main__':
    main()
