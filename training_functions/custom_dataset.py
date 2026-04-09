"""
@User: sandruskyi
CustomDataset to create a dataset object and to have the getitem function. Used in preparing_dataset.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

__all__ = ['CustomDataset', 'CustomDatasetNewLCRNExtend']

class CustomDatasetNewLCRNExtend(Dataset):
    def __init__(self, x_file, y_file, x_shape, y_shape, normalizations = False):
        self.x_file = x_file
        self.y_file = y_file
        self.x_shape = x_shape
        self.y_shape = y_shape

        self.length = x_shape[0]


    def __len__(self):
        return self.length

    def __getitem__(self, index):

        return self.x_data[index]/255.0 , self.y_data[index].float()

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        input_data = self.x[index]
        target = self.y[index]
        # Return a tuple containing the input data, target, and index
        return input_data, target, index

    def __len__(self):
        return len(self.y)
class CustomDataset_newCacophony(Dataset):
    def __init__(self, x, y, x_shape, y_shape):
        self.x = x
        self.y = y
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.length = x_shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        input_data = self.x[index]/255.0
        target = self.y[index].float()
        # Return a tuple containing the input data, target, and index
        return input_data, target, index

    def __len__(self):
        return self.length
"""
# Example usage
data = [...]  # your data samples
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=..., shuffle=...)

for inputs, targets, indices in dataloader:
    # Access the input data, target, and index within the loop
    print("Inputs:", inputs)
    print("Targets:", targets)
    print("Indices:", indices)
"""