# Author Alvaro Esperanca

from SVM import SVM
from PreProcessor import PreProcessor
from Validator import Validator
from GaussianKernel import GaussianKernel
import numpy as np


if __name__ == "__main__":
    pre = PreProcessor()
    val = Validator()
    clf = SVM(kernel=GaussianKernel(5.0), C=1.0)

    X_train, y_train = pre.loadTrainingSet("training_data/Cancer_Sample_Tokenized.csv")
    X_test = pre.loadTestSet("test_data/Tokenized_Cancer_test.csv")
    
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    validFile = open("validation_set/validation_set_labels.txt", "r")

    temp = validFile.readlines()
    validationLabels = [float(num) for num in temp]

    val.validate(validationLabels, predictions)

    val.report()

    # predFile = open("results/gaussian_kernel_s_5000_c_5.txt", "w")
    # statFile = open("results/gaussian_kernel_s_5000_c_5_stats.txt", "w")
    
    # for prediction in predictions:
    #     predFile.write("%d\n" % prediction)
    
    # statFile.write("%-20s %-5d\n" % ("True Positives:", val.truePositives()) )
    # statFile.write("%-20s %-5d\n" % ("True Negatives:", val.trueNegatives()) )
    # statFile.write("%-20s %-5d\n" % ("False Positives:", val.falsePositives()) )
    # statFile.write("%-20s %-5d\n\n" % ("False Negatives:", val.falseNegatives()) )
    # statFile.write("%-20s %.2f\n" % ("Accuracy:", val.precision()) )
    # statFile.write("%-20s %.2f\n" % ("Recall:", val.recall()) )

    # predFile.close()
    # statFile.close()