from __future__ import print_function # just in case you're using python2

from random import randint

NUMBER_OF_SHUFFLES = 100000

scores = {'star': [84, 57, 63, 99, 72, 46, 76, 91],
          'no_star': [81, 74, 56, 69, 66, 62, 69, 61, 87, 65, 44, 69]}

def mean(data_list):
    return sum(data_list)/len(data_list)

TEST_MEAN = mean(scores['star']) - mean(scores['no_star'])

averages = []

# Shuffling the 'star' test scores:
print("SH-SH-SH-SH-SHUFFLE!!!!!!")
for i in range(NUMBER_OF_SHUFFLES):
    dataset = scores["star"] + scores["no_star"]
    scores["star"] = []
    scores["no_star"] = []
    for score in dataset:
        j = randint(1,2)
        if j == 1:
            scores["star"].append(score)
        else:
            scores["no_star"].append(score)
    try:
        averages.append(mean(scores['star']) - mean(scores['no_star']))
    except:
        print("Note: Divide by zero encountered and escaped")

try:
    # Now, find the percentage of mean differences that are larger than our test:
    p_value = sum([1 for i in averages if i >= TEST_MEAN]) / len(averages)
    print("This is significant at the p = {} level".format(p_value))
except ZeroDivisionError as e:
    print("Division by zero error.\nPerhaps you're running your code before it's ready?")
