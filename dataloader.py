import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import copy


class seedlingFeaturesDataset(Dataset):
    def __init__(self, csv_file, root_dir):
            self.seedlingfeatures = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.seedlingfeatures)
    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        seedling_features = self.seedlingfeatures.iloc[idx, 0:3]
        quality = self.seedlingfeatures.iloc[idx, 3]

        seedling_features = np.array([seedling_features])
        quality = np.array([quality])

        sample = {'features':seedling_features, 'quality':quality}
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
