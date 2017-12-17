# Author Alvaro Esperanca

import numpy as np
import csv

class PreProcessor(object):
    def __init__(self):
        self.featureList = list()
        self.featureIndexMap = dict()

    def createDTM(self, documentSet):
        dtm = [[0 for x in range(len(self.featureList))] for y in range(len(documentSet))]
        
        for i in range(len(documentSet)):
            for feature in documentSet[i].split():
                if feature in self.featureList:
                    dtm[i][self.featureIndexMap[feature]] += 1

        return np.array(dtm)

    def loadTrainingSet(self, trainSetFilepath):
        trainSetFile = open(trainSetFilepath, "r")

        docList = list()
        labelList = list()

        documentReader = csv.DictReader(trainSetFile, delimiter='\t')
        
        for document in documentReader:
            docList.append(document["Text"])
            labelList.append(float(document["isClean"]))

        trainSetFile.close()

        for document in docList:
            for feature in document.split():
                if feature not in self.featureList:
                    self.featureList.append(feature)

        index = 0
        for feature in self.featureList:
            self.featureIndexMap[feature] = index
            index += 1

        X = self.createDTM(docList)
        y = np.array(labelList)

        return X, y

    def loadTestSet(self, testSetFilePath):
        testeSetFile = open(testSetFilePath, "r")
        docList = list()

        documentReader = csv.DictReader(testeSetFile, delimiter='\t')

        for document in documentReader:
            docList.append(document["Text"])

        X = self.createDTM(docList)

        return X