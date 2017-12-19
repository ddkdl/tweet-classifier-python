# Author: Alvaro Esperanca

from os import listdir
import sys, csv

def main(args):
    if len(args) != 1:
        print "incorrect number of arguments"
        return

    if args[0] != 'Asthma' and args[0] != 'Cancer' and args[0] != 'Diabetes':
        print "incorrect query"
        return

    query = args[0]

    years = {"2010" : list(), "2011" : list(), "2012" : list(), "2013" : list(), "2014" : list()}

    years["2010"] = listdir("./filtered_labels/" + query + "/2010/")
    years["2011"] = listdir("./filtered_labels/" + query + "/2011/")
    years["2012"] = listdir("./filtered_labels/" + query + "/2012/")
    years["2013"] = listdir("./filtered_labels/" + query + "/2013/")
    years["2014"] = listdir("./filtered_labels/" + query + "/2014/")

    for year in ["2010", "2011", "2012", "2013", "2014"]:
        for entry in years[year]:
            labelFile = open("./filtered_labels/" + query + "/" + year + "/" + entry, "r")
            
            T = 0
            F = 0
            
            for label in labelFile:
                if label == "1\n":
                    T += 1
                else:
                    F += 1
            
            outputFile = open("./filtered_labels/" + query + "/stats.csv", "a")

            csvWriter = csv.DictWriter(outputFile, fieldnames=["state", "year", "relevant", "irrelevant"])
            csvWriter.writerow({"state": entry, "year": year, "relevant": T, "irrelevant": F})

            outputFile.close()


if __name__ == '__main__':
    main(sys.argv[1:])