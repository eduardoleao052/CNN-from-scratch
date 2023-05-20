import numpy as np
import pandas as pd

class Augmenter():
    def __init__(self):
        pass

    def fit(self, data):
        self.data = np.array(data)

    def transform(self, data, data_y, dimensions = None, ratio = 4):
        data = np.array(data)
        data_y = np.array(data_y)
        if dimensions == None:
            dimensions = (int(np.sqrt(np.max(data.shape[1]))),int(np.sqrt(np.max(data.shape[1]))))
        new_data_x = []
        new_data_y = []
        for x, y in zip(data,data_y):
            data_reshaped = x.reshape(dimensions[0], dimensions[1])
            for k in range(ratio//4 + 1):
                data_expanded_x_pos = np.roll(data_reshaped, shift = k).reshape(-1,)
                data_expanded_x_neg = np.roll(data_reshaped, shift = -k).reshape(-1,)
                data_expanded_y_pos = np.roll(data_reshaped, shift = k, axis = 0).reshape(-1,)
                data_expanded_y_neg = np.roll(data_reshaped, shift = -k, axis = 0).reshape(-1,)
            new_data_x += [data_expanded_x_pos, data_expanded_x_neg, data_expanded_y_pos, data_expanded_y_neg]
            new_data_y += [y, y, y, y]
            #new_data_x += [data_expanded_x_pos, data_expanded_y_pos]
            #new_data_y += [y, y]

        df_x = pd.DataFrame(np.array(new_data_x))
        df_y = pd.DataFrame(np.array(new_data_y))
        return df_x, df_y
    
    def fit_transform(self, data, data_y, dimensions = None, ratio = 4):
        data = np.array(data)
        data_y = np.array(data_y)
        if dimensions == None:
                dimensions = (int(np.sqrt(np.max(data.shape[1]))),int(np.sqrt(np.max(data.shape[1]))))
        new_data_x = []
        new_data_y = []
        for x, y in zip(data,data_y):
            data_reshaped = x.reshape(dimensions[0], dimensions[1])
            for k in range(1, ratio//4 + 1):
                data_expanded_x_pos = np.roll(data_reshaped, shift = k).reshape(-1,)
                data_expanded_x_neg = np.roll(data_reshaped, shift = -k).reshape(-1,)
                data_expanded_y_pos = np.roll(data_reshaped, shift = k, axis = 0).reshape(-1,)
                data_expanded_y_neg = np.roll(data_reshaped, shift = -k, axis = 0).reshape(-1,)
                new_data_x += [data_expanded_x_pos, data_expanded_x_neg, data_expanded_y_pos, data_expanded_y_neg]
                new_data_y += [y, y, y, y]
                #new_data_x += [data_expanded_x_pos, data_expanded_y_pos]
                #new_data_y += [y, y]
        df_x = pd.DataFrame(np.array(new_data_x))
        df_y = pd.DataFrame(np.array(new_data_y))
        return df_x, df_y