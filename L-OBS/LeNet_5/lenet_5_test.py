import torch
import torch.utils.data
import torch.nn as nn

import torchvision
import torchvision.datasets
import numpy as np

from torchsummary import summary

trainset = torchvision.datasets.MNIST(root='./', train=True, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()]), download = False)
testset = torchvision.datasets.MNIST(root='./', train=False, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()]), download = False)

trainloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 100, shuffle = True)
testloader = torch.utils.data.DataLoader(dataset = testset, batch_size = 100, shuffle = True)


class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, 5),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(x.size(0), -1)
        output = self.fc(output)
        output = self.output(output)
        return output



n = LeNet_5()
n.load_state_dict(torch.load('params_lenet_5.pkl'))
# summary(n, (1, 32, 32))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        outputs = n(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))