self.tree

def predict(self, X_predict):
    # inputs X_predict (which is a dataframe)
    # returns a list with each item buinding the prediction for each row of the X_predict dataframe
    class_predictions = []
    node1 = self.tree
    node = self.tree
    for prediction in X_predict.iterrows():
        row = prediction[1]
        while len(node) > 1:
            if row[node[2][0]] <= node[2][1]:
                node = node[0]
            else:
                node = node[1]
        class_predictions.append(node)
        node = node1
    return class_predictions
