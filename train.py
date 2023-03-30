import torch.optim as optim
import torch
from torch import nn
from dataloader import seedlingFeaturesDataset, DataloaderBuilder
from model.seedlingnet import seedlingClassifier


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def train(model, dataloader, optimizer, criterion):
    for batch_idx, data in enumerate(dataloader):
        inputs = data['features'].type(torch.FloatTensor).squeeze()
        targets = data['quality'].type(torch.FloatTensor)

        outputs = model(inputs)

        #print(torch.cat((outputs,targets), dim=1))
        loss = criterion(outputs, targets)
        #acc = (outputs.reshape(-1).detach().numpy().round() == targets).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    seedling_dataset = seedlingFeaturesDataset(csv_file='./data/combined_file.csv', root_dir='./')
    ds = DataloaderBuilder(seedling_dataset)
    trainloader, testloader = ds.get_train_test_set(
        test_size=0.3,
        batch_size=20
    )

    # Hyperparameters of training
    learning_rate = 0.1
    epochs = 700
    running_loss = 0.0

    # MLP model applied to classify the seedlings
    model = seedlingClassifier(3)
    model.apply(init_weights)
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    criterion = nn.BCELoss()

    # training the MLP model
    model.train()
    for epoch in range(1000):
        train(model, trainloader, optimizer, criterion)

    # threshold of 40% gives the maximum precisionEntonces puedo concluir que el filtro de Kalman es un algoritmo que se usa en AI pero no es un algoritmo de AI
    threshold = 0.4

    correct = 0
    fail = 0

    # Testing on testset
    with torch.no_grad():
        model.eval()    
        for data in testloader:
            input = data['features'].type(torch.FloatTensor).squeeze()
            target = data['quality'].type(torch.FloatTensor).squeeze(1)
            
            output = model(input)

            if output.item() > threshold:
                pred = 1
            else:
                pred = 0
            
            if pred == target:
                correct += 1
            else:
                fail += 1

    print(correct/(correct+fail))