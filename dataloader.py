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

        quality = self.seedlingfeatures.iloc[idx, 0]
        seedling_features = pd.to_numeric(self.seedlingfeatures.iloc[idx, 1:4])
        fourier_descriptors = self.seedlingfeatures.iloc[idx, 4:41]
        #print(fourier_descriptors)
        #print(type(self.seedlingfeatures.iloc[idx, 3].apply(eval).apply(np.array)))
        #print(seedling_features.dtype)
        #print(pd.to_numeric(seedling_features).dtype)

        quality = np.array([quality])
        seedling_features = np.array([seedling_features])
        fourier_descriptors = np.array([fourier_descriptors])

        sample = {'quality':quality, 'features':seedling_features, 'fourier_descriptors': fourier_descriptors}
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
    #seedling_dataset = seedlingFeaturesDataset(csv_file='./data/combined_file.csv', root_dir='./')
    seedling_dataset = seedlingFeaturesDataset(csv_file='./data/combined_file_efd_added.csv', root_dir='./')

    ds = DataloaderBuilder(seedling_dataset)
    trainloader, testloader = ds.get_train_test_set(test_size=0.3,batch_size=16)

    # Hyperparameters of training
    learning_rate = 0.1
    epochs = 700
    running_loss = 0.0

    for data in testloader:
        print(data['features'])
        print(data['fourier_descriptors'])
        print(data['quality'])