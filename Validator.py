# Author Alvaro Esperanca

class Validator(object):
    def __init__(self):
        self.tp = 0.0
        self.tn = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def validate(self, validationLabels, testLabels):
        for i in range(len(validationLabels)):
            if validationLabels[i] == 1.0 and testLabels[i] == 1.0:
                self.tp += 1
            if validationLabels[i] == 1.0 and testLabels[i] == -1.0:
                self.fn += 1
            if validationLabels[i] == -1.0 and testLabels[i] == -1.0:
                self.tn += 1
            if validationLabels[i] == -1.0 and testLabels[i] == 1.0:
                self.fp += 1

    def truePositives(self):
        return self.tp

    def trueNegatives(self):
        return self.tn

    def falsePositives(self):
        return self.fp

    def falseNegatives(self):
        return self.fn

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)
    
    def report(self):
        print ("%-20s %-5d" % ("True Positives:", self.tp) )
        print ("%-20s %-5d" % ("True Negatives:", self.tn) )
        print ("%-20s %-5d" % ("False Positives:", self.fp) )
        print ("%-20s %-5d" % ("False Negatives:", self.fn) )

        print ("\n")

        print ("%-20s %.2f" % ("Accuracy:", self.precision()) )
        print ("%-20s %.2f" % ("Recall:", self.recall()) )