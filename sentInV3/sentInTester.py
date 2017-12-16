import json
import sys

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

# Will test insertSentences on the given dataList, using the given cost.
def testOn(insertSentences, dataList, cost, extractCostData):
    # Solve utf errors.
    reload(sys)
    sys.setdefaultencoding('utf8')

    # Go through data.
    predictedList = []
    sectionsTested = len(dataList) * 1.0
    sumSentencesPerSection = 0.0
    sentencesToInsert = 0.0
    sentencesCorrectlyInserted = 0.0
    sentencesToReject = 0.0
    sentencesCorrectlyRejected = 0.0
    numInsertedRightPlace = 0.0
    sumSentencesAway = 0.0
    for dataNum, data in enumerate(dataList):
        # Sanity check.
        if len(data["inputSection"]) < 5:
            continue

        # Call the algorithm.
        predictedSection = insertSentences(
            list(data["inputSentences"]),
            list(data["inputSection"]),
            cost,
            extractCostData(data))

        # Update metrics.
        sumSentencesPerSection += len(data["inputSection"])
        predictedIndexes = getInsertionIndexes(data["inputSentences"], predictedSection)
        sentencesAway = "N/A"
        for actualIndex, predictedIndex in zip(data["insertionIndexes"], predictedIndexes):
            if actualIndex is None:
                sentencesToReject += 1
                if predictedIndex is None:
                    sentencesCorrectlyRejected += 1
            else:
                sentencesToInsert += 1
                if predictedIndex is None:
                    continue
                sentencesCorrectlyInserted += 1
                if predictedIndex == actualIndex:
                    numInsertedRightPlace += 1
                sentencesAway = abs(predictedIndex - actualIndex)
                sumSentencesAway += sentencesAway

        # Show progress.
        if dataNum % 100 == 0:
            print "Ran {} sections".format(dataNum)
            sys.stdout.flush()

        # Save some info.
        savePredicted = False
        if savePredicted:
            data["predictedSection"] = predictedSection
            data["sentencesAway"] = sentencesAway
            del data["article"]
            predictedList.append(data)
            if dataNum % 100 == 99:
                with open("data/sentIn/sentInD2Predicted.json", 'ab') as outFile:
                    print "Dumping {} outputs".format(len(predictedList))
                    sys.stdout.flush()
                    outFile.write(json.dumps(predictedList, indent=4))
                    predictedList = []
    
    # Calculate some stats
    accuracy = (sentencesCorrectlyInserted + sentencesCorrectlyRejected) / (sentencesToInsert + sentencesToReject)
    precision = sentencesCorrectlyInserted / (sentencesToReject - sentencesCorrectlyRejected + sentencesCorrectlyInserted)
    recall = sentencesCorrectlyInserted / sentencesToInsert
    f1 = 2 * (precision * recall) / (precision + recall)

    # Print and return stats.
    print (
        "\n-------- Results --------\n" + 
        "sectionsTested: {0:.0f}\n" +
        "avgInsertionPoints: {1:.4f}\n" + 
        "sentencesToInsert: {2:.0f}\n" +
        "sentencesCorrectlyInserted: {3:.0f}\n" +
        "sentencesToReject: {4:.0f}\n" +
        "sentencesCorrectlyRejected: {5:.0f}\n" +
        "numInsertedRightPlace: {6:.0f}\n" +
        "avgSentencesAway: {7:.4f}\n" +
        "accuracy: {8: .4f}\n" +
        "precision: {9: .4f}\n" +
        "recall: {10: .4f}\n" +
        "f1: {11: .4f}\n"
    ).format(
        sectionsTested,
        sumSentencesPerSection / sectionsTested + 1,
        sentencesToInsert,
        sentencesCorrectlyInserted,
        sentencesToReject,
        sentencesCorrectlyRejected,
        numInsertedRightPlace,
        sumSentencesAway / sentencesCorrectlyInserted,
        accuracy,
        precision,
        recall,
        f1
    )


    return {
        "sectionsTested": sectionsTested,
        "avgInsertionPoints": sumSentencesPerSection / sectionsTested + 1,
        "sentencesToInsert": sentencesToInsert,
        "sentencesCorrectlyInserted": sentencesCorrectlyInserted,
        "sentencesToReject": sentencesToReject,
        "sentencesCorrectlyRejected": sentencesCorrectlyRejected,
        "numInsertedRightPlace": numInsertedRightPlace,
        "avgSentencesAway": sumSentencesAway / sentencesCorrectlyInserted,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
