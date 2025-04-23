import random
from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)

class ScrappyKNN():
    def fit(self, train_feature, train_label):
        self.train_feature = train_feature
        self.train_label=train_label

    def predict(self, test_feature):
        prediction = []
        for row in test_feature:
            label = self.closest(row)
            prediction.append(label)
        return prediction

    def closest(self, row):
        best_dist = euc(row, self.train_feature[0])
        best_index = 0
        for i in range(1, len(self.train_feature)):
            dist = euc(row, self.train_feature[i])
            if dist < best_dist:
                best_dist = dist
                best_index = 1
        return self.train_label[best_index]