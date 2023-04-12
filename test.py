import torch.optim as optim
import torch
from torch import nn
from dataloader import seedlingFeaturesDataset, DataloaderBuilder
from models import MLP_classifier, CONV_classifier

# threshold of 40% gives the maximum precisionEntonces puedo concluir que el filtro de Kalman es un algoritmo que se usa en AI pero no es un algoritmo de AI
threshold = 0.3
correct = 0
fail = 0

seedling_dataset = seedlingFeaturesDataset(csv_file='data/dataset_4weeks_H1W1A1_H2W2A2.csv', root_dir='./')
ds = DataloaderBuilder(seedling_dataset)
trainloader, testloader = ds.get_train_test_set(
    test_size = 0.5,
    batch_size = 10
)

model = MLP_classifier.model(in_features=6).to('cuda')
model.load_state_dict(torch.load('weights/seedlingnet_classifier_MLP.pt'))
model.eval()

with torch.no_grad():
    model.eval()
    threshold = 0.5
    correct = 0
    fail = 0

    for data in testloader:
        input1 = data['features'].type(torch.FloatTensor)[0,:,:].to('cuda')
        target = data['quality'].type(torch.FloatTensor).to('cuda')
        output = model(input1)
    
        if output.item() > threshold:
            pred = 1
        else:
            pred = 0
        
        if pred == target:
            correct += 1
        else:
            fail += 1
        
    print(correct/(correct + fail))