import torch
import torch.utils.data
import torch.nn as nn

import torchvision
import torchvision.datasets
import numpy as np

num_epochs = 100
learning_rate = 0.001


# 1. load dataset
trainset = torchvision.datasets.MNIST(root='./', train=True, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()]), download = False)
testset = torchvision.datasets.MNIST(root='./', train=False, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()]), download = False)

trainloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 100, shuffle = True)
testloader = torch.utils.data.DataLoader(dataset = testset, batch_size = 100, shuffle = True)


# 2. build LeNet-5
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

net = LeNet_5()
print('net already')
para = list(net.parameters())
print(para)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)


total_step = len(trainloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):

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
torch.save(net.state_dict(), 'params_lenet_5.pkl')