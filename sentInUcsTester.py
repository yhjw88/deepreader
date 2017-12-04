import json
import sys
import ucs

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

def main():
    # Solve utf errors.
    reload(sys)
    sys.setdefaultencoding('utf8')

    # Load data.
    filename = "data/sentInDev.json"
    dataList = []
    with open(filename, 'rb') as inFile:
        dataList = json.load(inFile)

    # Algorithm takes too long, use on less data.
    dataList = dataList[:100]

    # Go through data.
    sectionsToTest = len(dataList) * 1.0
    sumSentencesPerSection = 0.0
    numToInsert = 0.0
    numInserted = 0.0
    numInsertedCorrectly = 0.0
    sumSentencesAway = 0.0
    for dataNum, data in enumerate(dataList):
        # Call the algorithm.
        ucs.corpus = data["section"] + data["sentences"]
        actualSection = ucs.InsertSentences(list(data["sentences"]), list(data["section"]), ucs.reward)
        
        # Update metrics.
        sumSentencesPerSection += len(data["section"])
        actualIndexes = getInsertionIndexes(data["sentences"], actualSection)
        for expectedIndex, actualIndex in zip(data["insertionIndexes"], actualIndexes):
            numToInsert += 1
            if actualIndex is None:
                continue

            numInserted += 1
            if actualIndex == expectedIndex:
                numInsertedCorrectly += 1
            sumSentencesAway += abs(expectedIndex - actualIndex)

        # Show progress.
        if dataNum % 10 == 0:
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
        #     for sentence in actualSection:
        #         outFile.write(sentence)
        #         outFile.write("\n")

    print (
        "sectionsTested: {}, avgInsertionPoints: {}, sentencesToInsert: {}, " +
        "numActuallyInserted: {}, avgInsertedCorrectly: {}, avgSentencesAway: {}").format(
            sectionsToTest,
            sumSentencesPerSection / sectionsToTest + 1,
            numToInsert,
            numInserted,
            numInsertedCorrectly / numInserted,
            sumSentencesAway / numInserted
        )

if __name__ == "__main__":
    main()
