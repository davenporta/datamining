# This file is obsolete. Please see code-new.py. Thanks! - Tony & Gherardo

# Standard import statements:
from itertools import combinations, chain
from statistics import mean, median

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

% matplotlib inline

plt.style.use("ggplot")

# your class code here

data2 = [["human", "warm-blooded", "yes", "no", "no", "yes"],
     ["pigeon", "warm-blooded", "no", "no", "no", "no"],
     ["elephant", "warm-blooded", "yes", "yes", "no", "yes"],
     ["leopard shark", "cold-blooded", "yes", "no", "no", "no"],
     ["turtle", "cold-blooded", "no", "yes", "no", "no"],
     ["penguin", "cold-blooded", "no", "no", "no", "no"],
     ["eel", "cold-blooded", "no", "no", "no", "no"],
     ["dolphin", "warm-blooded", "yes", "no", "no", "yes"],
     ["spiny anteater", "warm-blooded", "no", "yes", "yes", "yes"],
     ["gila monster", "cold-blooded", "no", "yes", "yes", "no"]]

data = [["human", 1, 1, 0, 0, 1],
     ["pigeon", 1, 0, 0, 0, 0],
     ["elephant", 1, 1, 1, 0, 1],
     ["leopard shark", 0, 1, 0, 0, 0],
     ["turtle", 0, 0, 1, 0, 0],
     ["penguin", 0, 0, 0, 0, 0],
     ["eel", 0, 0, 0, 0, 0],
     ["dolphin", 1, 1, 0, 0, 1],
     ["spiny anteater", 1, 0, 1, 1, 1],
     ["gila monster", 0, 0, 1, 1, 0]]

df = pd.DataFrame(data, columns = ["Name", 
                                "Body Temperature", 
                                "Gives Birth", 
                                "Four-legged", 
                                "Hibernates", 
                                "Class Label"])
y_dat = df["Class Label"]
X_dat = df[["Body Temperature","Gives Birth", "Four-legged", "Hibernates"]]



def giniCalc(p):
    """p is a list of classes, such as ['S', 'NS', 'NS', 'NS', 'S'] or [0,1,0,2,1,0] if 0,1,2 are the things 
    that corresponds to an increasing list of some parameter that you want to split by."""
    classes = list(set(p))  # find all the classes
    gini = 1
    for i in classes:
        gini -= (float(p.count(i)) / len(p)) ** 2
    return gini


def splitter(node_x, node_y):  # feed me dataframes
    i_old = giniCalc(list(node_y))
    poss_splits = []
    for p in node_x.columns:
        set_list = list(node_x[p].unique())
        set_list.sort()
        # print(set_list)
        for t in set_list[:-1]:
            # print("split at: %s" % split)
            downside = list(node_y[node_x[p] <= t])
            upside = list(node_y[node_x[p] > t])
            # print(list(downside))
            # print(list(upside))
            g = i_old - (
                (len(downside) / len(node_y)) * giniCalc(downside) + (len(upside) / len(node_y)) * giniCalc(upside))
            poss_splits.append((p, t, g))
    return poss_splits  # i'll spit out list of tuples with:
    # (predictor, threshhold *split at less than or equal to this number*, information gain)


def find_best_split(possible_splits):
    g = 1000
    p = 0
    t = 0
    for split in possible_splits:
        if split[2] < g:
            p = split[0]
            t = split[1]
            g = split[2]
    return p, t, g


# method that given a set of predictors returns the classes


# node format: Node ID, Left Child ID, Right, Child ID, Split Parameter, Split Data

tree  # declaration of nonstatic variable --> TODO find type for this variable

# method that given a set of predictors returns the classes


tree  # declaration of nonstatic variable --> TODO find type for this variable


def fit(X, y, threshold=0, n=10):
    """ n is the limit for the number of nodes the tree will have
    predictors/class are dataframes with same indexing
        t is the threshold value at which the split is occurring
        X and y are the dataframes
    """

    if n == 0:
        return None

    n -= 1

    node = [[None], [None], [None]]
    predictor, threshold, info_gain = find_best_split(X, y)

    if info_gain == 1:  # if the node is pure, then stop
        return None

    class_column = y.column

    # data = X.merge(y)
    combined_data = pd.concat([X, y], axis=1)

    # left = data[data[predictor] <= threshold]
    # right = data[data[predictor] > threshold]
    left = combined_data.loc[combined_data[predictor] <= threshold]
    right = combined_data.loc[combined_data[predictor] > threshold]

    node[0] = fit(left[predictor], left[class_column], t=threshold)
    node[1] = fit(right[predictor], right[class_column], t=threshold)
    node[2] = threshold

    return node  # returns a list with left node, right node, threshold
