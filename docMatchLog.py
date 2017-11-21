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

    # Average empirical loss.
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
            if i % 10000 == 0:
                print 'Iterations: {}, Grad: {}, Theta: {}'.format(i, np.linalg.norm(grad, 1), theta)
                sys.stdout.flush()
                if i > 500000:
                    break
            if np.linalg.norm(prev_theta - theta) < 1e-15:
                print('Converged in %d iterations' % i)
                break
        self.theta = theta
        return theta

    def getPredicted(self, testX):
        testX = self.addIntercept(testX)
        scores = np.dot(testX, self.theta)
        return [1 if score >= 0 else -1 for score in scores]

    def plot(self, filename="data/plot.png"):
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
        plt.xlabel("x_1")
        plt.ylabel("x_2")
        plt.savefig(filename)
        return

def main():
    # Do regression.
    logisticRegression = LogisticRegression(np.loadtxt("data/docMatchTrainX.txt"), np.loadtxt("data/docMatchTrainY.txt"))
    logisticRegression.doWork(10)
    logisticRegression.plot()

    # Test accuracy
    testX = np.loadtxt("data/docMatchTestX.txt")
    testY = np.loadtxt("data/docMatchTestY.txt")
    predictedY = logisticRegression.getPredicted(testX)
    count = 0
    for actY, preY in zip(testY, predictedY):
        if actY == preY:
            count += 1
    print "Num correct: {} Num total: {}".format(count, len(testY))

    # Plot it for fun.
    logisticRegression.resetData(np.loadtxt("data/docMatchTestX.txt"), np.loadtxt("data/docMatchTestY.txt"))
    logisticRegression.plot("data/plot2.png")

if __name__ == '__main__':
    main()
