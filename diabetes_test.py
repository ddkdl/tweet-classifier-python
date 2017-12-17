# Author Alvaro Esperanca

from SVM import SVM
from PreProcessor import PreProcessor
from Validator import Validator
from LinearKernel import LinearKernel
from PolynomialKernel import PolynomialKernel
from GaussianKernel import GaussianKernel
import numpy as np


if __name__ == "__main__":
    pre = PreProcessor()

    # Setting up training dataset
    X_train, y_train = pre.loadTrainingSet("training_data/Diabetes_Sample_Tokenized.csv")
    X_test = pre.loadTestSet("test_data/Tokenized_Diabetes_test.csv")
    

    # Setting up validation labels
    validFile = open("validation_set/diabetes_validation_labels.txt", "r")

    temp = validFile.readlines()
    validationLabels = [float(num) for num in temp]

    validFile.close()
    
    # Setting up different kernels and C's
    kernel_list = list()
    c_list = list()
    sigma_list = list()

    c_list.append(1)
    c_list.append(5)
    c_list.append(10)
    c_list.append(50)
    c_list.append(100)
    c_list.append(1000)
    c_list.append(None)

    sigma_list.append(0.0005)
    sigma_list.append(0.5)
    sigma_list.append(1.0)
    sigma_list.append(2.5)
    sigma_list.append(5.0)
    sigma_list.append(5000.0)

    # Testing linear kernel
    for c in c_list:
        clf = SVM(kernel=LinearKernel(), C=c)
        val = Validator()
        
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        val.validate(validationLabels, predictions)

        predFile = open("results/diabetes_linear_kernel_c_" + str(c) + ".txt", "w")
        statFile = open("results/diabetes_linear_kernel_c_" + str(c) + "_stats.txt", "w")
    
        for prediction in predictions:
            predFile.write("%d\n" % prediction)
        
        statFile.write("%-20s %-5d\n" % ("True Positives:", val.truePositives()) )
        statFile.write("%-20s %-5d\n" % ("True Negatives:", val.trueNegatives()) )
        statFile.write("%-20s %-5d\n" % ("False Positives:", val.falsePositives()) )
        statFile.write("%-20s %-5d\n\n" % ("False Negatives:", val.falseNegatives()) )
        statFile.write("%-20s %.2f\n" % ("Accuracy:", val.precision()) )
        statFile.write("%-20s %.2f\n" % ("Recall:", val.recall()) )

        predFile.close()
        statFile.close()

    # Testing polynomial kernel
    for c in c_list:
        clf = SVM(kernel=PolynomialKernel(), C=c)

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        val.validate(validationLabels, predictions)

        predFile = open("results/diabetes_polynomial_kernel_c_" + str(c) + ".txt", "w")
        statFile = open("results/diabetes_polynomial_kernel_c_" + str(c) + "_stats.txt", "w")
    
        for prediction in predictions:
            predFile.write("%d\n" % prediction)
        
        statFile.write("%-20s %-5d\n" % ("True Positives:", val.truePositives()) )
        statFile.write("%-20s %-5d\n" % ("True Negatives:", val.trueNegatives()) )
        statFile.write("%-20s %-5d\n" % ("False Positives:", val.falsePositives()) )
        statFile.write("%-20s %-5d\n\n" % ("False Negatives:", val.falseNegatives()) )
        statFile.write("%-20s %.2f\n" % ("Accuracy:", val.precision()) )
        statFile.write("%-20s %.2f\n" % ("Recall:", val.recall()) )

        predFile.close()
        statFile.close()

    # Testing gaussian kernel
    for sigma in sigma_list:
        for c in c_list:
            clf = SVM(kernel=GaussianKernel(sigma), C=c)

            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)

            val.validate(validationLabels, predictions)

            predFile = open("results/diabetes_gaussian_kernel_s_" + str(sigma) + "_c_" + str(c) + ".txt", "w")
            statFile = open("results/diabetes_gaussian_kernel_s_" + str(sigma) + "_c_" + str(c) + "_stats.txt", "w")
        
            for prediction in predictions:
                predFile.write("%d\n" % prediction)
            
            statFile.write("%-20s %-5d\n" % ("True Positives:", val.truePositives()) )
            statFile.write("%-20s %-5d\n" % ("True Negatives:", val.trueNegatives()) )
            statFile.write("%-20s %-5d\n" % ("False Positives:", val.falsePositives()) )
            statFile.write("%-20s %-5d\n\n" % ("False Negatives:", val.falseNegatives()) )
            statFile.write("%-20s %.2f\n" % ("Accuracy:", val.precision()) )
            statFile.write("%-20s %.2f\n" % ("Recall:", val.recall()) )

            predFile.close()
            statFile.close()