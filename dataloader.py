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
            # This transform is used in experiments
            self.transform = T.Compose([
                T.Resize(size = (224,224)),
                T.ToTensor(),
                ThresholdTransform(thr_255=60)
            ])

    def __len__(self):
        return len(self.seedlingfeatures)
    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        quality = self.seedlingfeatures.iloc[idx, 0]
        seedling_features = pd.to_numeric(self.seedlingfeatures.iloc[idx, 1:7])
        
        vertical_mask_path = self.seedlingfeatures.iloc[idx, 7]
        vertical_mask = Image.open(vertical_mask_path)
        vertical_mask = ImageOps.grayscale(vertical_mask)

        horizontal_mask_path = self.seedlingfeatures.iloc[idx, 8]
        horizontal_mask = Image.open(horizontal_mask_path)
        horizontal_mask = ImageOps.grayscale(horizontal_mask)

        if self.transform:
            horizontal_mask = self.transform(horizontal_mask)
            vertical_mask = self.transform(vertical_mask)
            
        #fourier_descriptors = self.seedlingfeatures.iloc[idx, 4:41]

        quality = np.array([quality])
        seedling_features = np.array([seedling_features])
        #fourier_descriptors = np.array([fourier_descriptors])

        sample = {'quality':quality, 
                  'features':seedling_features, 
                  'horizontal_mask': horizontal_mask, 
                  'vertical_mask':vertical_mask} #, 'fourier_descriptors': fourier_descriptors}
        
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

        self.input = None
        self.pred = None
    
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
    seedling_dataset = seedlingFeaturesDataset(csv_file='data/dataset.csv', root_dir='./')

    ds = DataloaderBuilder(seedling_dataset)
    trainloader, testloader = ds.get_train_test_set(test_size=0.3,batch_size=16)

    for data in testloader:        
        print(data['features'])
        print(data['vertical_mask'])
        print(data['quality'])