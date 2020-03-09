import pandas as pd
import numpy as np

class PartitionData():
    
    def __init__(self, data_path, training_size):
        self.data_path = data_path
        self.training_size = training_size
        self.temp_data = self.split_data()
        
    def split_data(self):
        data = open(self.data_path, 'r', encoding="ISO-8859-1")    
        df = pd.DataFrame(data)
        train, val = np.split(df, [int(self.training_size*len(df))])
        # create a temp file
        with open("../data/temp_train.txt", "w") as file:
            for line in train.values.tolist():
                file.write(line[0])                        
        return train.values.tolist()
    
    def get_data(self):
        return self.temp_data