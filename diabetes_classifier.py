# Author: Alvaro Esperanca
from os import listdir
from SVM import SVM
from PreProcessor import PreProcessor
from GaussianKernel import GaussianKernel

def main():
    
    pre = PreProcessor()
    clfA = SVM(kernel=GaussianKernel(5.0), C=1)
    clfB = SVM(kernel=GaussianKernel(1.0), C=1)

    # Loading training data
    X_train, y_train = pre.loadTrainingSet("./training_data/Diabetes_Sample_Tokenized.csv")

    # Training classifiers
    clfA.fit(X_train, y_train)
    clfB.fit(X_train, y_train)

    # Loading raw tweets
    years = {"2010" : list(), "2011" : list(), "2012" : list(), "2013" : list(), "2014" : list()}

    years["2010"] = listdir("./data/Diabetes/2010/")
    years["2011"] = listdir("./data/Diabetes/2011/")
    years["2012"] = listdir("./data/Diabetes/2012/")
    years["2013"] = listdir("./data/Diabetes/2013/")
    years["2014"] = listdir("./data/Diabetes/2014/")

    # Classifying tweets
    for year in ["2010", "2011", "2012", "2013", "2014"]:
        for entry in years[year]:
            print "Classifying %s %s" % (year, entry)
            
            X = pre.loadTestSet("./data/Diabetes/" + year + "/" + entry)
            predictionsA = clfA.predict(X)
            predictionsB = clfB.predict(X)
            
            outputFile = entry.split(".")[0] + ".txt"
            
            predFile = open("./filtered_labels/Diabetes/" + year + "/" + outputFile, "w")

            for i in range( len(predictionsA) ):
                if predictionsA[i] == -1 and predictionsB[i] == -1:
                    prediction = -1
                else:
                    prediction = 1

                predFile.write("%d\n" % prediction)

            predFile.close()

if __name__ == '__main__':
    main()