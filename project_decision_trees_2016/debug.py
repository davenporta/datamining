from itertools import combinations, chain
from statistics import mean, median

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, r2_score


def read_file(file_name):
    # Reads file. Designed for noisy_dataset.csv
    # Input: filename
    # Returns: X, y: dataframes
    dataframe = pd.read_csv(file_name, index_col=0)
    X = dataframe[['x_1', 'x_2']]
    y = dataframe['y']
    return X, y


def fit(X, y, n=3, branch=True):
    print("Running fit")
    # Inputs:
    # n : the limit for the number of nodes the tree will have predictors/class are dataframes with same indexing
    # X and y: dataframes
    # Returns: a list with left node, right node, threshold

    if n == 0:
        # print("done")
        print("node is:", None)
        return set(y.values)

    n -= 1
    node = [[], [], []]

    # if branch:
    #     print("Right branch")
    # else:
    #     print("Left branch")

    predictor, threshold, info_gain, gini = find_best_split(X, y)
    print("gini" + str(gini))

    # if calculate_gini(list(y)) == 0:
    #     # print(y)
    #     return None

    if gini < 0.000001:  # if the node is pure (gini impurity is 0), then stop
        print("node is:", None)
        return set(y.values)

    y = pd.DataFrame({'y': list(y)})  # in order to get the name of the single column dataframe
    class_columns = list(y.columns.values)

    combined_data = pd.concat([X, y], axis=1)
    pred_columns = [x for x in combined_data.columns.values if x not in class_columns]
    # print("pred columns", pred_columns)

    left = combined_data[combined_data[predictor] <= threshold]  # .reset_index().drop("index", axis=1)
    right = combined_data[combined_data[predictor] > threshold]  # .reset_index().drop("index", axis=1)

    node[0] = fit(left[pred_columns], left[class_columns], n=n, branch=True)
    node[1] = fit(right[pred_columns], right[class_columns], n=n, branch=False)
    node[2] = threshold

    print("node is:", node)
    return node


def find_best_split(node_x, node_y):
    # Input: X and Y Dataframes
    # Outputs: Best split:
    # tuple with (predictor, threshold *split at less than or equal to this number*, information gain)
    # Relies on: get_possible_splits(), calculate_gini()
    # print(node_x)
    # print(node_y)
    possible_splits = get_possible_splits(node_x, node_y)
    # print(possible_splits)
    ig = -1
    p = 0
    t = 0
    g = 0
    for split in possible_splits:
        if split[2] > ig:  # this has to be greater not less than!
            # print(split[2])
            p = split[0]
            t = split[1]
            ig = split[2]
            g = split[3]
    return p, t, ig, g


def get_possible_splits(node_x, node_y):
    # Input: X and Y Dataframes
    # Returns: All possible splits along with corresponding information gain:
    # list of tuples with:(predictor, threshold *split at less than or equal to this number*, information gain)
    # Used by: find_best_split()
    # Relies on: calculate_gini()
    # print("getting possible splits")
    # print(list(node_y.values.flatten()))
    # print(node_x)
    gp = calculate_gini(list(node_y.values.flatten()))
    # print("g = ", gp)
    poss_splits = []
    for pred in node_x.columns:
        set_list = list(node_x[pred].unique())
        set_list.sort()
        # print(set_list)
        for t in set_list[:-1]:
            # print("split at: %s" % split)
            downside = list(node_y[node_x[pred] <= t].values.flatten())
            upside = list(node_y[node_x[pred] > t].values.flatten())
            # print("list(downside)", list(downside))
            # print("list(downside)", list(upside))
            ig = gp - (
                (len(downside) / len(node_y)) * calculate_gini(downside)
                + (len(upside) / len(node_y)) * calculate_gini(upside))
            # print("appending", (pred, t, ig, gp))
            poss_splits.append((pred, t, ig, gp))
    # print("poss splits:", poss_splits)
    return poss_splits  # i'll spit out list of tuples with:
    # (predictor, threshhold *split at less than or equal to this number*, information gain, and the impurity)


def calculate_gini(P):
    # Input: a list of classes, such as ['S', 'NS', 'NS', 'NS', 'S'] or [0,1,0,2,1,0] if 0,1,2 are the things
    # that corresponds to an increasing list of some parameter that you want to split by.
    # Returns: gini score.
    # Used by: get_possible_splits().
    classes = list(set(P))  # find all the classes
    # print("p = ", P)
    # print("calculate gini")
    # print("classes:", classes)
    gini = 1
    # print("gini initiated as 1")
    for i in classes:
        gini -= (float(P.count(i)) / len(P)) ** 2
        # print("gini changed to:", gini)
    # print("returning gini=", gini)
    return gini


# X, y = read_file("noisy_dataset.csv")
# print(fit(X, y))
# print(find_best_split(X, y))

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

data1 = [["human", 1, 1, 0, 0, 1],
         ["pigeon", 1, 0, 0, 0, 0],
         ["elephant", 1, 1, 1, 0, 1],
         ["dolphin", 1, 1, 0, 0, 1],
         ["spiny anteater", 1, 0, 1, 1, 1]]

data2 = [["leopard shark", 0, 1, 0, 0, 0],
         ["turtle", 0, 0, 1, 0, 0],
         ["penguin", 0, 0, 0, 0, 0],
         ["eel", 0, 0, 0, 0, 0],
         ["gila monster", 0, 0, 1, 1, 0]]

df = pd.DataFrame(data, columns=["Name",
                                 "Body Temperature",
                                 "Gives Birth",
                                 "Four-legged",
                                 "Hibernates",
                                 "Class Label"])
y_dat = df["Class Label"]
X_dat = df[["Body Temperature", "Gives Birth", "Four-legged", "Hibernates"]]

# print(X_dat)
# print(y_dat)

print(fit(X_dat, y_dat))
# print(find_best_split(X_dat, y_dat))
