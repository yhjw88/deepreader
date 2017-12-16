import matplotlib.pyplot as plt
import numpy as np

def plotCats(X, Y, filename):
    plt.figure()

    # Split the list of x's.
    x1Zeroes = []
    x2Zeroes = []
    x1Ones = []
    x2Ones = []
    x1Twos = []
    x2Twos = []
    for xRow, y in zip(X, Y):
        if y == 0:
            x1Zeroes.append(xRow[0])
            x2Zeroes.append(xRow[1])
        elif y == 1:
            x1Ones.append(xRow[0])
            x2Ones.append(xRow[1])
        else:
            x1Twos.append(xRow[0])
            x2Twos.append(xRow[1])

    # Now plot the things.
    plt.plot(x1Zeroes, x2Zeroes, "g+")
    plt.plot(x1Ones, x2Ones, "b+")
    plt.plot(x1Twos, x2Twos, "r+")
    plt.xlabel("title")
    plt.ylabel("text")
    plt.savefig(filename)
    return

def main():
    devX = np.loadtxt("data/docMatchIITrainX.txt")
    devY = np.loadtxt("data/docMatchIITrainY.txt")
    plotCats(devX, devY, "data/docMatchIITrain.png")

if __name__ == '__main__':
    main()
