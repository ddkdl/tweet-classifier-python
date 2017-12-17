import csv, operator, sys


def main(args):
    if len(args) == 0 or len(args) > 1:
        print "No input file specified"
        return

    input_file_path = args[0] 
    output_file_path = input_file_path.rsplit('.', 1)[0] + "_dtm.txt"
    output_header_file = input_file_path.rsplit('.', 1)[0] + "_header.txt"

    # Opening file as input
    in_file = open(input_file_path, "r")

    # defining the lists and map required
    tweet_list = list()
    word_list = list()
    word_index_map = dict()

    # Opening csv reader for the file
    tweet_reader = csv.DictReader(in_file, delimiter='\t')

    # Adding the text in each tweet to the tweet list
    for row in tweet_reader:
        tweet_list.append(row["Text"])

    in_file.close()

    # Creating a word list from all the word occurences in the tweet list
    for tweet in tweet_list:
        for word in tweet.split():
            if word not in word_list:
                word_list.append(word)


    index = 0

    # Creating the map (i.e. pseudo hash function) of words to indices
    for word in word_list:
        word_index_map[word] = index
        index = index + 1

    tweet_count = len(tweet_list)
    word_count = len(word_list)

    # Creating a 2d matrix filled with zeros
    dtm = [[0 for x in range(word_count)] for y in range(tweet_count)]

    # Adding the number of occurences of a word in each tweet
    for i in range(tweet_count):
        for word in tweet_list[i].split():
            dtm[i][word_index_map[word]] += 1

    # Sort the word map by index
    sorted_map = sorted(word_index_map.items(), key=operator.itemgetter(1))

    # Creating a separate header file
    with open(output_header_file, "w") as header_file:
        for item in sorted_map:
            header_file.write("%s " % item[0])

    # Output DTM to a file
    with open(output_file_path, "w") as o_file:
        o_file.write("%d %d\n"%(tweet_count, word_count))
        for row in dtm:
            for item in row:
                o_file.write("%d " % item)
            o_file.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])