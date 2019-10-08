class Tree:
    def __init__(self, x=None):
        self.tree = x
    
    def fit(self, X, y, n=50):
        self.tree = self.fitter(X, y, n)
        return
    
    def predict(self, X_predict):
        class_predictions = []
        for prediction in X_predict:
            while node[2] != None:
                if prediction[node[3]] <= node[2]:
                    node = node[0]
                else:
                    node = node[1]
            class_predictions.append(node[3])
        return class_predictions    
    
    def calculate_gini(self, P):
        # Input: a list of classes, such as ['S', 'NS', 'NS', 'NS', 'S'] or [0,1,0,2,1,0] if 0,1,2 are the things
        # that corresponds to an increasing list of some parameter that you want to split by.
        # Returns: gini score.
        # Used by: get_possible_splits().
        classes = list(set(P))  # find all the classes
        gini = 1
        for i in classes:
            gini -= (float(P.count(i)) / len(P)) ** 2
        return gini

    def find_best_split(self, node_x, node_y):
        gp = self.calculate_gini(list(node_y.values.flatten()))
        poss_splits = []
        for pred in node_x.columns:
            set_list = list(node_x[pred].unique())
            set_list.sort()
            for t in set_list[:-1]:
                downside = list(node_y[node_x[pred] <= t].values.flatten())
                upside = list(node_y[node_x[pred] > t].values.flatten())
                #print(list(downside))
                #print(list(upside))
                ig = gp - (
                    (len(downside) / len(node_y)) * self.calculate_gini(downside)
                    + (len(upside) / len(node_y)) * self.calculate_gini(upside))
                poss_splits.append((pred, t, ig, gp))
        #print(poss_splits)
        try:
            return max(poss_splits, key=itemgetter(2))
        except:
            return None

    def fitter(self, X, y, n, branch=True):
        #print("Running fit")
        # Inputs:
        # n : the limit for the number of nodes the tree will have predictors/class are dataframes with same indexing
        # X and y: dataframes
        # Returns: a list with left node, right node, threshold

        if n == 0:
            print("No more levels")
            return list(y.unique())

        n -= 1
        node = [[], [], []]
        tup = self.find_best_split(X, y)

        if tup != None:
            predictor, threshold, info_gain, gini = tup
        else:
            return list(y.unique())

        if gini <= 0.0001:  # if the node is pure (gini impurity is 0), then stop
            return list(y.unique())

        node[0] = self.fitter(X[X[predictor] <= threshold], y[X[predictor] <= threshold], n=n, branch=True)
        node[1] = self.fitter(X[X[predictor] > threshold], y[X[predictor] > threshold], n=n, branch=False)
        node[2] = (predictor,threshold)

        return node
    
    def __str__(self):
        # This is what comes out when you print.
        return "Tree with x = {}".format(self.x.__repr__())
