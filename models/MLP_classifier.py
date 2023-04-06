import torch.optim as optim
import torch
from torch import nn
from tqdm import tqdm

class model(nn.Module):
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


def train(model, dataloader, optimizer, criterion):
    for batch_idx, data in enumerate(tqdm(dataloader)):
        inputs1 = data['features'].type(torch.FloatTensor).squeeze()
        targets = data['quality'].type(torch.FloatTensor)
        outputs = model(inputs1)
        loss = criterion(outputs, targets)        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(loss)

def validation(model, dataloader, threshold):
    correct = 0
    fail = 0
    
    for data in dataloader:
        input1 = data['features'].type(torch.FloatTensor)[0,:,:]
        #input2 = data['fourier_descriptors'].type(torch.FloatTensor)
        target = data['quality'].type(torch.FloatTensor)
        #print(input1.shape,input2.shape)
        output = model(input1)
        #print('target:', target, '| output:', output)

        if output.item() > threshold:
            pred = 1
        else:
            pred = 0
        
        if pred == target:
            correct += 1
        else:
            fail += 1
        
    return correct/(correct + fail)