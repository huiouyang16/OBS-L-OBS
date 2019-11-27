import torch
import torch.utils.data
import torch.nn as nn

import torchvision
import torchvision.datasets
import numpy as np


testset = torchvision.datasets.MNIST(root = './', train = False, transform = torchvision.transforms.ToTensor(), download = False)
testloader = torch.utils.data.DataLoader(dataset = testset, batch_size = 100, shuffle = True)


class N(nn.Module):
    def __init__(self):
        super(N, self).__init__()
        self.input_layer = nn.Linear(784, 300)
        self.hidden_layer = nn.Linear(300, 100)
        self.output_layer = nn.Linear(100, 10)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.relu(x)
        x = self.hidden_layer(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x


n = N()
n.load_state_dict(torch.load('pruned2.pkl'))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images = images.reshape(-1, 28 * 28)
        outputs = n(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
