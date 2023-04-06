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
        #self.conv2 = nn.Conv1d(32, 64, 7)
        #self.conv3 = nn.Conv1d(64, 32, 7)
        self.conv4 = nn.Conv1d(32, 16, 7)
        self.conv5 = nn.Conv1d(16, 8, 7)
        self.adaptative_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        #x = torch.relu(self.conv2(x))
        #x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.adaptative_avg_pool(x)
        return x


class SeedlingNet(nn.Module):
    def __init__(self, n_encoder_features=None, n_verbose_features=None, use_hiden_features = False):
        super().__init__()
        self.use_hiden_features = use_hiden_features
        if use_hiden_features != False:
            self.classifer = seedlingClassifier(in_features=n_verbose_features + n_encoder_features)
            self.encoder = seedlingFeatureEncoder()
        else:
            self.classifer = seedlingClassifier(in_features=n_verbose_features)

    def forward(self, input):
        bbox = None
        fourier = None
        image = None

        for keyword in input:
            horizontal = input[keyword]['vertical']
            vertical = input[keyword]['horizontal']
            if keyword == 'bbox' and horizontal is not None and vertical is not None:    
                bbox = torch.cat([horizontal,vertical], dim=2)    
            if keyword == 'fourier' and horizontal is not None and vertical is not None:    
                fourier = torch.cat([horizontal,vertical], dim=1)
            if keyword == 'image' and horizontal is not None and vertical is not None:
                image = torch.cat([horizontal, vertical], dim=1)
        
        
        # vertical_hwa = input["vertical_hwa"]
        # horizontal_hwa = input["horizontal_hwa"]
        # vertical_fourier_descriptors = input["vertical_fourier_descriptors"]
        # horizontal_fourier_descriptors = input["horizontal_fourier_descriptors"]
        # horizontal_image = input["horizontal_image"]
        # vertical_image = input["vertical_image"]

        
        # if vertical_hwa != None and horizontal_hwa != None:
        #     bbox_features = torch.cat([vertical_hwa, horizontal_hwa], dim=2)
        
        # if vertical_fourier_descriptors != None and horizontal_fourier_descriptors != None:
        #     fourier_features = torch.cat([vertical_fourier_descriptors, horizontal_fourier_descriptors], dim=2)


        #verbose_features = features_HWA #torch.flatten(features_HWA, start_dim=1)
        # print(verbose_features.shape)
        # if self.use_hiden_features != False:
        #     hidden_features = torch.flatten(self.encoder(fourier_descriptors), start_dim=1)
        #     features = torch.cat([verbose_features, hidden_features], dim=1)
        # else:
        #     features = verbose_features

        # class_seelding = self.classifer(features)        
        return


def test_model():
    model = SeedlingNet(n_verbose_features = 4, use_hiden_features=False)
    x1 = torch.rand([20,1,4])
    x2 = torch.rand([20,1,37])
    x3 = torch.rand([20,1,224,224])
    input = {
        "bbox": {'horizontal':x1, 'vertical':x1},
        "fourier": {'horizontal':x2, 'vertical':x2},
        "image":{'horizontal':x3, 'vertical':x3}
    }
    y = model(input)
    #print(y)

if __name__ == '__main__':
    test_model()