import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import copy
from PIL import Image, ImageOps
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt


class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data type

class Invert():
  def __call__(self, x):
    return torchvision.transforms.functional.invert(x)  # do not change the data type


class seedlingFeaturesDataset(Dataset):
    def __init__(self, csv_file, root_dir):
            self.seedlingfeatures = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.seedlingfeatures)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()        

        sample = {'length': self.seedlingfeatures.iloc[idx,0], 
                  'vertical_area': self.seedlingfeatures.iloc[idx,1],
                  'vertical_height': self.seedlingfeatures.iloc[idx,2],
                  'vertical_width': self.seedlingfeatures.iloc[idx,3],
                  'horizontal_area': self.seedlingfeatures.iloc[idx,4],
                  'horizontal_height': self.seedlingfeatures.iloc[idx,5],
                  'horizontal_width': self.seedlingfeatures.iloc[idx,6]
                  } 
  
        return sample

    def setSeedlingFeatures(self, idx):
        self.seedlingfeatures = self.seedlingfeatures.iloc[idx]
        return

    def get_subset(self, idx):
        dataset_copy = copy.copy(self)
        dataset_copy.setSeedlingFeatures(idx)
        return dataset_copy


class DataloaderBuilder():
    def __init__(self, dataset):

        self.dataset = dataset
        self.len = len(dataset)
        self.range = range(0, self.len)
    
    def get_train_test_set(self, test_size, batch_size=1):
        Xtrain_idx, Xtest_idx, Ytrain_idx, Ytest_idx = train_test_split(self.range, 
                                                        self.range, 
                                                        test_size=test_size, 
                                                        random_state=42,
                                                        shuffle=True)
        
        train_loader = DataLoader(self.dataset.get_subset(Xtrain_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.dataset.get_subset(Xtest_idx), shuffle=True)
        return train_loader, test_loader


if __name__ == '__main__':
    seedling_dataset = seedlingFeaturesDataset(csv_file='dataset/data.csv', root_dir='./')
    ds = DataloaderBuilder(seedling_dataset)
    trainloader, testloader = ds.get_train_test_set(test_size=0.3, batch_size=10)
    for data in trainloader:
        print(data['length'])
        print(data['vertical_area'])