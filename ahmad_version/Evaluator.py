# evaluator = Evaluator(y_pred.squeeze(), y_test)
# precision = evaluator.get_Precision()
# f1 = evaluator.get_f1_score()
# del evaluator

import torch
import sys
import time


class Evaluator():
    """
    Class to handle all evaluation functions
    """

    def __init__(self, y_pred, y_actual, converted_y_pred=None):
        super(Evaluator, self).__init__()

        # actual is tensor
        self.actual_labels = y_actual

        if(converted_y_pred):
            self.predicted_labels = converted_y_pred
        else:
            self.predicted_labels = self._convertToClassIndices(y_pred)

        self.TruePositives = self._getTruePositives()
        self.FalsePositives = self._getFalsePositives()
        self.FalseNegatives = self._getFalseNegatives()

        self.Precision = self.get_Precision_score()
        self.Recall = self.get_Recall_score()

    def _convertToClassIndices(self, y_vector):
        y_labels = []
        for y in y_vector:
            pred = values, indices = torch.max(y, 0)
            # get class index array
            y_labels.append(indices.item())
        return y_labels

    def get_Precision(self):
        assert len(self.actual_labels) == len(self.predicted_labels)
        correct = 0
        for idx, pred in enumerate(self.predicted_labels):
            if pred == self.actual_labels[idx].item():
                correct += 1.0
        precision = (correct)/(len(self.actual_labels)) * 100
        return correct, precision

    def get_Precision_score(self):
        return self.TruePositives / (self.TruePositives + self.FalsePositives)

    def get_Recall_score(self):
        return self.TruePositives / (self.TruePositives + self.FalseNegatives)

    # – F1 = 2PR/(P+R)
    def get_f1_score(self):
        return (2 * self.Precision * self.Recall) / (self.Precision + self.Recall)

    # micro averaging
    def _getTruePositives(self):
        assert len(self.actual_labels) == len(self.predicted_labels)
        count = 0
        for idx, pred in enumerate(self.predicted_labels):
            if pred == self.actual_labels[idx].item():
                count += 1.0
        return count

    # micro averaging
    def _getFalsePositives(self):
        assert len(self.actual_labels) == len(self.predicted_labels)
        count = 0
        for idx, pred in enumerate(self.predicted_labels):
            if pred != self.actual_labels[idx].item():
                count += 1.0
        return count

    # micro averaging
    def _getFalseNegatives(self):
        assert len(self.actual_labels) == len(self.predicted_labels)
        count = 0
        for idx, actual in enumerate(self.actual_labels.tolist()):
            if actual != self.predicted_labels[idx]:
                count += 1.0
        return count


if __name__ == "__main__":
    # python3 Evaluator.py
    print("Evaluator says Hello World!")
    print("Welcome to the Evaluator Tester")

    # 10 elements, 3 classes
    # y_predicted = [1, 2, 3, 2, 3, 1, 3, 2, 1, 2]
    # y_actual = torch.tensor([2, 2, 3, 1, 3, 1, 3, 2, 3, 2])

    y_predicted = [0, 5, 7, 1, 8, 8, 7, 1, 7, 5, 10, 7, 5, 3, 5, 5, 9, 8, 7, 9, 6, 10, 1, 8, 3, 7, 8, 3, 8, 9, 9, 1, 10, 1, 4, 2, 4, 5, 7, 5, 0, 3, 3, 1, 2, 6, 3, 6, 1,
                   5, 9, 9, 8, 6, 5, 10, 3, 0, 5, 9, 2, 6, 1, 0, 4, 8, 6, 5, 2, 9, 9, 0, 4, 10, 0, 1, 1, 4, 3, 6, 10, 1, 5, 6, 2, 8, 9, 10, 6, 3, 1, 9, 5, 8, 9, 5, 4, 7, 6, 2]

    y_actual = torch.tensor([5, 6, 2, 10, 9, 5, 1, 4, 0, 6, 3, 1, 3, 7, 6, 7, 3, 9, 3, 0, 0, 6, 3, 3, 5, 2, 2, 4, 9, 3, 3, 3, 8, 8, 3, 8, 10, 1, 8, 7, 3, 2, 0, 9, 3, 1, 2, 9, 9,
                             7, 5, 0, 7, 1, 3, 9, 10, 9, 8, 2, 4, 10, 8, 3, 9, 2, 8, 1, 9, 4, 7, 2, 6, 8, 7, 1, 8, 10, 2, 10, 8, 10, 9, 4, 0, 3, 3, 9, 10, 0, 0, 3, 0, 7, 0, 3, 6, 5, 2, 8])

    evaluator = Evaluator(y_predicted, y_actual, converted_y_pred=y_predicted)
    correct_count, precision1 = evaluator.get_Precision()
    precision2 = evaluator.Precision * 100
    recall = evaluator.Recall * 100
    f1 = evaluator.get_f1_score()
    del evaluator

    print("Correct predictions: {} / {}".format(correct_count, len(y_predicted)))
    print("Precision: {}".format(precision1))
    print("Precision 2: {}".format(precision2))
    print("Recall: {}".format(recall))
    print("F1 Micro: {}".format(f1))
