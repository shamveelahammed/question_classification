import torch
import numpy as np
import pandas as pd


# conMat = ConfusionMatrix(y_predicted, y_actual)
# conMat.getConfusionMatrix()

# https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62


class ConfusionMatrix():
    def __init__(self, classes, y_pred, y_actual):
        super(ConfusionMatrix, self).__init__()

        self.predictedData = y_pred
        self.y_actual = y_actual.tolist()
        self.classes = range(0, len(classes))

    def getConfusionMatrix(self):
        classes = []
        result = []

        # go through classes vertically (predicted)
        for class_predicted in self.classes:
            countList = []
            # go through classes horizontally (actual)
            for class_actual in self.classes:
                # go though data
                count = 0
                for idx, pred in enumerate(self.predictedData):
                    if pred == class_predicted and self.y_actual[idx] == class_actual:
                        count += 1
                countList.append(count)
            result.append(countList)

        result = np.asarray(result)

        print(pd.DataFrame(result))


if __name__ == "__main__":
    # python3 Evaluator.py
    print("Evaluator says Hello World!")
    print("Welcome to Confusion Matrix Generator!")

    # 10 elements, 3 classes
    y_predicted = [1, 2, 3, 2, 3, 1, 3, 2, 1, 2]
    y_actual = torch.tensor([2, 2, 3, 1, 3, 1, 3, 2, 3, 2])

    conMat = ConfusionMatrix(y_predicted, y_actual)
    conMat.getConfusionMatrix()

# [[1 1 1]
#  [1 3 0]
#  [0 0 3]]
