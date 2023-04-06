import torch.optim as optim
import torch
from torch import nn
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.resnet50 = resnet50(weights=weights, progress=False).eval()
        self.encoder = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.transforms = weights.transforms()
    
    def forward(self,x):
        with torch.no_grad():
            return self.encoder(x)


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.hidden1 = nn.Linear(2048, 128)
        self.hidden2 = nn.Linear(2048, 128)
        self.classiy = nn.Linear(256, 1)

    def forward(self, x):
        x1 = self.encoder(x['horizontal_mask']).view(-1, 2048)
        x2 = self.encoder(x['vertical_mask']).view(-1, 2048)
        x = torch.cat([self.hidden1(x1), self.hidden2(x2)], dim=1)
        return torch.sigmoid(self.classiy(x))


def train(model, dataloader, optimizer, criterion):
    for batch_idx, data in enumerate(tqdm(dataloader)):
        horizontal_mask = data['horizontal_mask'].type(torch.FloatTensor)
        vertical_mask = data['vertical_mask'].type(torch.FloatTensor)

        input = {
            'horizontal_mask': torch.cat([horizontal_mask, horizontal_mask, horizontal_mask], dim=1),
            'vertical_mask': torch.cat([vertical_mask, vertical_mask, vertical_mask], dim=1)
        }
        output = model(input)
        target = data['quality'].type(torch.FloatTensor)
        
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('loss:',loss.item())


def validation(model, dataloader, threshold):
    correct = 0
    fail = 0
    
    for data in dataloader:
        horizontal_mask = data['horizontal_mask'].type(torch.FloatTensor)
        vertical_mask = data['vertical_mask'].type(torch.FloatTensor)

        input = {
            'horizontal_mask': torch.cat([horizontal_mask, horizontal_mask, horizontal_mask], dim=1),
            'vertical_mask': torch.cat([vertical_mask, vertical_mask, vertical_mask], dim=1)
        }
        output = model(input)
        target = data['quality'].type(torch.FloatTensor)
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


if __name__ == '__main__':
    encoder = model()
    encoder.eval()
    
    x = {
        'horizontal_mask': torch.rand([1,3,224,224]),
        'vertical_mask': torch.rand([1,3,224,224])
    }
    
    y = encoder(x)
    print(y.shape)
