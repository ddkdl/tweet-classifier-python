import csv, operator

# Opening file as input
training_set = open("training_data/Cancer_Sample_Tokenized.csv", "r")

# defining the lists and map required
tweet_list = list()
word_list = list()

# Opening csv reader for the file
tweet_reader = csv.DictReader(training_set, delimiter='\t')

# Adding the text in each tweet to the tweet list
for row in tweet_reader:
    tweet_list.append(row["Text"])

training_set.close()

# Creating a word list from all the word occurences in the tweet list
for tweet in tweet_list:
    for word in tweet.split():
        if word not in word_list:
            word_list.append(word)

with open("training_data/training_set_features.txt", "r") as ofile:
    for item in word_list:
        ofile.write()