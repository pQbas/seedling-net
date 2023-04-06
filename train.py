import torch.optim as optim
import torch
from torch import nn
from dataloader import seedlingFeaturesDataset, DataloaderBuilder
from tqdm import tqdm
#from model.seedlingnet import SeedlingNet


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':
    #seedling_dataset = seedlingFeaturesDataset(csv_file='./data/combined_file.csv', root_dir='./')
    seedling_dataset = seedlingFeaturesDataset(csv_file='data/dataset.csv', root_dir='./')

    ds = DataloaderBuilder(seedling_dataset)
    trainloader, testloader = ds.get_train_test_set(
        test_size = 0.3,
        batch_size = 10
    )
    
    # Hyperparameters of training
    learning_rate = 0.001
    epochs = 50
    running_loss = 0.0

    # MLP model applied to classify the seedlings
    from models import MLP_classifier, CONV_classifier
    
    #model = MLP_classifier.model(in_features=6)
    model = CONV_classifier.model()

    model.apply(init_weights)
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    criterion = nn.BCELoss()

    # training the MLP model
    model.train()
    for epoch in range(epochs):
        #MLP_classifier.train(model, trainloader, optimizer, criterion)
        CONV_classifier.train(model, trainloader, optimizer, criterion)
        

    # threshold of 50% gives the maximum precisionEntonces puedo concluir que el filtro de Kalman es un algoritmo que se usa en AI pero no es un algoritmo de AI
    with torch.no_grad():
        model.eval()
        #acc = MLP_classifier.validation(model, testloader, threshold=0.5)
        acc = CONV_classifier.validation(model, testloader, threshold=0.5)
        print('Accuracy:', acc)