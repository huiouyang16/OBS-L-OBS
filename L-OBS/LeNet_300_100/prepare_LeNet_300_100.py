import torch
import torch.utils.data
import torch.nn as nn

import torchvision
import torchvision.datasets
import numpy as np

num_epochs = 100
learning_rate = 0.001


# 1. load dataset
trainset = torchvision.datasets.MNIST(root = './', train = True, transform = torchvision.transforms.ToTensor(), download = False)
testset = torchvision.datasets.MNIST(root = './', train = False, transform = torchvision.transforms.ToTensor(), download = False)

trainloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 100, shuffle = True)
testloader = torch.utils.data.DataLoader(dataset = testset, batch_size = 100, shuffle = True)


# 2. build LeNet-300-100
class LeNet_300_100(nn.Module):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
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


net = LeNet_300_100()
print('net already')
para = list(net.parameters())
print(para)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)


total_step = len(trainloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        images = images.reshape(-1, 28*28)

        outputs = net(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


para = list(net.parameters())
print(para)
torch.save(net.state_dict(), 'params_lenet_300_100.pkl')