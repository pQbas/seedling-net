import torch.optim as optim
import torch
from torch import nn


class seedlingClassifier(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


class seedlingFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7)
        self.conv2 = nn.Conv1d(32, 64, 7)
        self.conv3 = nn.Conv1d(64, 32, 7)
        self.conv4 = nn.Conv1d(32, 16, 7)
        self.conv5 = nn.Conv1d(16, 8, 7)
        self.adaptative_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.adaptative_avg_pool(x)
        return x



class SeedlingNet(nn.Module):
    def __init__(self, use_hiden_features = False):
        super().__init__()
        self.use_hiden_features = use_hiden_features

        if use_hiden_features != False:
            self.classifer = seedlingClassifier(in_features=12)
            self.encoder = seedlingFeatureEncoder()
        else:
            self.classifer = seedlingClassifier(in_features=4)

    def forward(self, features_HWA, fourier_descriptors):
        verbose_features = torch.flatten(features_HWA, start_dim=1)
        
        if self.use_hiden_features != False:
            hidden_features = torch.flatten(self.encoder(fourier_descriptors), start_dim=1)
            features = torch.cat([verbose_features, hidden_features], dim=1)
        else:
            features = verbose_features
        class_seelding = self.classifer(features)
        
        return class_seelding


def test_model():
    
    model = SeedlingNet(use_hiden_features=False)
    x1 = torch.rand([10,1,4])
    x2 = torch.rand([10,1,35])
    y = model(x1, x2)
    print(y)

if __name__ == '__main__':
    test_model()