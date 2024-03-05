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
    seedling_dataset = seedlingFeaturesDataset(csv_file='dataset/data.csv', root_dir='./')

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
    from models import mlp_classifier, conv_classifier
    
    model = mlp_classifier.model(in_features=6)

    model.apply(init_weights).to('cuda')
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    criterion = nn.BCELoss()

    # training the MLP model
    model.train()
    for epoch in tqdm(range(epochs)):

        for batch_idx, data in enumerate(trainloader):
            targets = (data['length']>0.5).type(torch.FloatTensor).to('cuda')

            inputs = torch.cat([data['vertical_area'][...,None].type(torch.FloatTensor),
                                 data['vertical_height'][...,None].type(torch.FloatTensor),
                                 data['vertical_width'][...,None].type(torch.FloatTensor),
                                 data['horizontal_area'][...,None].type(torch.FloatTensor),
                                 data['horizontal_height'][...,None].type(torch.FloatTensor),
                                 data['horizontal_width'][...,None].type(torch.FloatTensor)]
                                 ,
                                 dim = 1).to('cuda')
            
            outputs = model(inputs).squeeze()   
            loss = criterion(outputs, targets)      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(loss.item())

    # testing
    model.eval()
    threshold = 0.5
    correct = 0
    fail = 0

    for data in testloader:
        targets = (data['length']>0.5).type(torch.FloatTensor).to('cuda')
        
        inputs = torch.cat([data['vertical_area'][...,None].type(torch.FloatTensor),
                            data['vertical_height'][...,None].type(torch.FloatTensor),
                            data['vertical_width'][...,None].type(torch.FloatTensor),
                            data['horizontal_area'][...,None].type(torch.FloatTensor),
                            data['horizontal_height'][...,None].type(torch.FloatTensor),
                            data['horizontal_width'][...,None].type(torch.FloatTensor)]
                            ,
                            dim = 1).to('cuda')
                            
        outputs = model(inputs).squeeze()   

        if outputs.item() > threshold:
            pred = 1
        else:
            pred = 0
        
        if pred == targets:
            correct += 1
        else:
            fail += 1
    
    print(correct/(correct + fail))

    # to save the model
    torch.save(model.state_dict(), f'weights/seedlingnet_classifier_MLP.pt')