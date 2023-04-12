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
    seedling_dataset = seedlingFeaturesDataset(csv_file='data/dataset_4weeks_H1W1A1_H2W2A2.csv', root_dir='./')

    ds = DataloaderBuilder(seedling_dataset)
    trainloader, testloader = ds.get_train_test_set(
        test_size = 0.3,
        batch_size = 10
    )
    
    # Hyperparameters of training
    learning_rate = 0.01
    epochs = 500
    running_loss = 0.0

    # MLP model applied to classify the seedlings
    from models import MLP_classifier, CONV_classifier
    
    model = MLP_classifier.model(in_features=6)
    #model = CONV_classifier.model()

    model.apply(init_weights).to('cuda')
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    criterion = nn.BCELoss()

    # training the MLP model
    model.train()
    for epoch in tqdm(range(epochs)):
        #MLP_classifier.train(model, trainloader, optimizer, criterion)
        #CONV_classifier.train(model, trainloader, optimizer, criterion)

        for batch_idx, data in enumerate(trainloader):
            inputs1 = data['features'].type(torch.FloatTensor).squeeze().to('cuda')
            targets = data['quality'].type(torch.FloatTensor).to('cuda')
            outputs = model(inputs1)
            #print(outputs.shape)
            #print(targets.shape)
            #print(str(torch.cat([outputs,targets],dim=1)))
            loss = criterion(outputs, targets)      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(loss.item())

    # threshold of 50% gives the maximum precisionEntonces puedo concluir que el filtro de Kalman es un algoritmo que se usa en AI pero no es un algoritmo de AI
    with torch.no_grad():
        model.eval()
        # acc = MLP_classifier.validation(model, testloader, threshold=0.5)
        # #acc = CONV_classifier.validation(model, testloader, threshold=0.5)
        # print('data/dataset.csvAccuracy:', acc)
        threshold = 0.5
        correct = 0
        fail = 0

        for data in testloader:
            input1 = data['features'].type(torch.FloatTensor)[0,:,:].to('cuda')
            #input2 = data['fourier_descriptors'].type(torch.FloatTensor)
            target = data['quality'].type(torch.FloatTensor).to('cuda')
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
            
        print(correct/(correct + fail))

    # to save the model
    torch.save(model.state_dict(), f'weights/seedlingnet_classifier_MLP.pt')