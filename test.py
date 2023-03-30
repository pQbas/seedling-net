import torch.optim as optim
import torch
from torch import nn


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