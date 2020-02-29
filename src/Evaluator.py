# evaluator = Evaluator(y_pred.squeeze(), y_test)
# precision = evaluator.get_Precision()
# del evaluator

import torch
import sys
import time


class Evaluator():
    """
    Class to handle all evaluation functions
    """

    def __init__(self, y_pred, y_actual):
        super(Evaluator, self).__init__()

        #actual is tensor
        self.actual_labels = y_actual

        self.predicted_labels = self._convertToClassIndices(y_pred)

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

    # def get_f_score
