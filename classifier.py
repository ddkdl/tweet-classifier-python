# Author: Alvaro Esperanca
from os import listdir
from SVM import SVM
from PreProcessor import PreProcessor

def gaussian_kernel(x, y, sigma=5000.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def main():
    dirList = list()

    temp = listdir("./data/Cancer/2010/")
    dirList.append(temp)
    temp = listdir("./data/Cancer/2011/")
    dirList.append(temp)
    temp = listdir("./data/Cancer/2012/")
    dirList.append(temp)
    temp = listdir("./data/Cancer/2013/")
    dirList.append(temp)
    temp = listdir("./data/Cancer/2014/")
    dirList.append(temp)

    pre = PreProcessor()
    clf = SVM(kernel=gaussian_kernel, C=1.0)

    X_train, y_train = pre.loadTrainingSet("training_data/Cancer_Sample_Tokenized.csv")    
    clf.fit(X_train, y_train)

    

if __name__ == '__main__':
    main()