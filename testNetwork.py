'''
    NICE!
    It works as requested!
    
    This is a test to make sure CUDA Works on some neural network code I found on PyTorch.
    NOT MY CODE!!!!
'''

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('We are using: ', device)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512,10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

print('\n\n')
X = torch.rand(1,28,28,device=device)

# calling model(X) calls the forward + background operations
# never call forward directly!. something w/ inheritance and stuff

logits = model(X)
pred_prob = nn.Softmax(dim=1)(logits)
y_pred = pred_prob.argmax(1)
print('Prediction: ', y_pred)

