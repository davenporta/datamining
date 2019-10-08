import pandas as pd

def read_file(file_name):
    # Reads file. Designed for noisy_dataset.csv
    # Input: filename
    # Returns: X, y: dataframes
    dataframe = pd.read_csv(file_name, index_col=0)
    X = dataframe[['x_1', 'x_2']]
    y = dataframe['y']
    return X, y


def fit(X, y, n=3, branch=True):
    # Inputs:
    # n : the limit for the number of nodes the tree will have predictors/class are dataframes with same indexing
    # X and y: dataframes
    # Returns: a list with left node, right node, threshold

    if n == 0:
        print("done")
        return None

    n -= 1
    node = [[None], [None], [None]]

    if(branch):
        print("Right branch")
    else:
        print("Left branch")

    predictor, threshold, info_gain, gini = find_best_split(X, y)
    print("gini" + str(gini))

    # if calculate_gini(list(y)) == 0:
    #     # print(y)
    #     return None

    if gini == 0:  # if the node is pure (gini impurity is 0), then stop
        return None

    y = pd.DataFrame({'y': list(y)}) # in order to get the name of the single column dataframe
    class_columns = list(y.columns.values)

    combined_data = pd.concat([X, y], axis=1)
    pred_columns = [x for x in combined_data.columns.values if x not in class_columns]

    left = combined_data[combined_data[predictor] <= threshold]#.reset_index().drop("index", axis=1)
    right = combined_data[combined_data[predictor] > threshold]#.reset_index().drop("index", axis=1)

    node[0] = fit(left[pred_columns], left[class_columns], n=n, branch=True)
    node[1] = fit(right[pred_columns], right[class_columns], n=n, branch=False)
    node[2] = threshold

    return node


def find_best_split(node_x, node_y):
    # Input: X and Y Dataframes
    # Outputs: Best split:
    # tuple with (predictor, threshold *split at less than or equal to this number*, information gain)
    # Relies on: get_possible_splits(), calculate_gini()
    print(node_x)
    print(node_y)
    possible_splits = get_possible_splits(node_x, node_y)
    print(possible_splits)
    ig = -1
    p = 0
    t = 0
    g = 0
    for split in possible_splits:
        if split[2] > ig: # this has to be greater not less than!
            print(split[2])
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
    gp = calculate_gini(list(node_y.values.flatten()))
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

## DATA TESTING
# Dataset 1
from sklearn.model_selection import train_test_split
X = [["human", "warm-blooded", "yes", "no", "no", "yes"],
     ["pigeon", "warm-blooded", "no", "no", "no", "no"],
     ["elephant", "warm-blooded", "yes", "yes", "no", "yes"],
     ["leopard shark", "cold-blooded", "yes", "no", "no", "no"],
     ["turtle", "cold-blooded", "no", "yes", "no", "no"],
     ["penguin", "cold-blooded", "no", "no", "no", "no"],
     ["eel", "cold-blooded", "no", "no", "no", "no"],
     ["dolphin", "warm-blooded", "yes", "no", "no", "yes"],
     ["spiny anteater", "warm-blooded", "no", "yes", "yes", "yes"],
     ["gila monster", "cold-blooded", "no", "yes", "yes", "no"]]

def binarize(var):
    if var == "no":
        return 0
    elif var == "yes":
        return 1
df[["Gives Birth","Four-legged","Hibernates","Class Label"]] = df[["Gives Birth","Four-legged","Hibernates","Class Label"]].applymap(binarize)
body_temp = {"warm-blooded": 1, "cold-blooded": 0}
df['Body Temperature'] = df['Body Temperature'].map(body_temp)
df.head()
X_mtrain, X_mtest, y_mtrain, y_mtest = train_test_split(df[df.columns.tolist()[1:len(df.columns) - 1]], df["Class Label"], test_size=.3)


# Dataset 2
from sklearn.datasets import load_iris
dataset = load_iris()
X_iris = pd.DataFrame(data=dataset['data'])
X_iris.columns = dataset['feature_names']
y_iris = pd.DataFrame(data=dataset['target'])
y_iris.columns = ['Class']
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=.3)

# Dataset 3
from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples = 150, n_features = 5, centers = 3)
X_blob = pd.DataFrame(data=X)
y_blob = pd.DataFrame(data=y)
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, test_size=.3)

# Dataset 4
df_titanic_train = pd.read_csv("train_titanic.csv")
df_titanic_train.head()
del df_titanic_train["PassengerId"]
del df_titanic_train["Name"]
del df_titanic_train["Cabin"]
del df_titanic_train["Ticket"]

gender_bin = {"male": 1, "female": 0}
df_titanic_train['Sex'] = df_titanic_train['Sex'].map(gender_bin)
print(df_titanic_train['Embarked'].unique())

df_titanic_train[df_titanic_train['Embarked'].isin(['S','C','Q']) == False]

df_titanic_train = df_titanic_train.drop(61)
df_titanic_train = df_titanic_train.drop(829)
df_titanic_train = df_titanic_train.reset_index(drop=True)
df_titanic_train.head()

def change_embark(dataframe):
    cherbourg = []
    southampton = []
    for x in dataframe['Embarked']:
        if x == 'S':
            southampton.append(1)
            cherbourg.append(0)
        if x == 'C':
            southampton.append(0)
            cherbourg.append(1)
        if x == 'Q':
            southampton.append(0)
            cherbourg.append(0)
    dataframe['is_southampton'] = southampton
    dataframe['is_cherbourg'] = cherbourg

change_embark(df_titanic_train)
del df_titanic_train["Embarked"]
df_titanic_train.head()

df_titanic_train[df_titanic_train['Age'].isnull()]

median = df_titanic_train['Age'].median()
df_titanic_train['Age'] = df_titanic_train['Age'].fillna(median)

df_titanic_train['Fare'].hist(bins = 20)
plt.show()
df_titanic_train['Age'].hist(bins = 20)
plt.show()

df_titanic_train[df_titanic_train['Fare'] < 1]

class_one = df_titanic_train[df_titanic_train.Pclass == 1].Fare.median()
class_two = df_titanic_train[df_titanic_train.Pclass == 2].Fare.median()
class_three = df_titanic_train[df_titanic_train.Pclass == 3].Fare.median()
indices = df_titanic_train[df_titanic_train['Fare'] < 1].index
for x in indices:
    if df_titanic_train.loc[x, 'Pclass'] == 1:
        df_titanic_train.loc[x, 'Fare'] = class_one
    if df_titanic_train.loc[x, 'Pclass'] == 2:
        df_titanic_train.loc[x, 'Fare'] = class_two
    if df_titanic_train.loc[x, 'Pclass'] == 3:
        df_titanic_train.loc[x, 'Fare'] = class_three
        
X_titan = df_titanic_train[df_titanic_train.columns.tolist()[1:]]
y_titan = df_titanic_train['Survived']
X_titanic_train, X_titanic_test, y_titanic_train, y_titanic_test = train_test_split(X_titan, y_titan, test_size=.3)

# dataset 5
dataframe = pd.read_csv("noisy_dataset.csv", index_col = 0)
X_noisy = dataframe[['x_1', 'x_2']]
y_noisy = dataframe['y']
X_noisy_train, X_noisy_test, y_noisy_train, y_noisy_test = train_test_split(X_noisy, y_noisy, test_size=.3)
