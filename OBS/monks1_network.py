"""
    This code is used to define the monks1 network,
    train/load the trained the monks1 network,
    calculate/load the hessian inv matrix of trained the monks1 network,
    pruned/save the monks1 network,
    and test the accuracy

    author: Hui Ouyang
"""
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import os
########################################################################################################################
#   Config: Begin here
########################################################################################################################
# re train the xor network or not if the network is not exit in the file
re_train = False
# number of training iteration
epoch = 390
# re compute the inverse of hessian matrix or not if the hessian is not exit in the file
re_hessian_inv = False
# display the whole pruning process or not (used to find the best pruning threshold)
display_process = False
# the threshold of the pruning process, only work when it is not in the display process mode
threshold = 1.05e-6
# the initial alpha (alpha inverse in fact) for the hessian matrix (1e4 to 1e8 in generally)
alpha = 1000000
########################################################################################################################
#   Config: End here
########################################################################################################################


# define the monk1 model
# structure: 17 - Linear - sigmoid - 3 - Linear - sigmoid - 1
class MONK1Network(nn.Module):

    def __init__(self):
        super(MONK1Network, self).__init__()
        self.fc1 = nn.Linear(17, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# define the dataset
class MONKDataset(data.Dataset):
    def __init__(self, file_name):
        file = open(file_name, 'r')
        samples = []
        labels = []
        for line in file:
            sample = [0 for _ in range(17)]
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            sample[int(words[1]) - 1] = 1
            sample[int(words[2]) + 2] = 1
            sample[int(words[3]) + 5] = 1
            sample[int(words[4]) + 7] = 1
            sample[int(words[5]) + 10] = 1
            sample[int(words[6]) + 14] = 1
            samples.append(sample)
            labels.append(int(words[0]))
        self.samples = torch.from_numpy(np.array(samples)).type(torch.FloatTensor)
        self.labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)

    def __getitem__(self, index):
        sample, label = self.samples[index], self.labels[index]
        return sample, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':

    # get a monk1 model
    monk1_network = MONK1Network()

    # get the train set and test set
    train_set = MONKDataset('/../dataset/monks-problems/monks-1.train')
    train_loader = data.DataLoader(train_set, shuffle=True)
    print('train set already')
    test_set = MONKDataset('/../dataset/monks-problems/monks-1.test')
    test_loader = data.DataLoader(test_set, shuffle=True)
    print('test set already')
    print()

    # train or load the network
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(monk1_network.parameters(), lr=0.1, momentum=0.9)
    if not os.path.exists('monk1_model/trained/monk1_network.pkl') or re_train:
        for epoch in range(epoch):

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the input
                inputs, labels = data
                labels = labels.unsqueeze(1)

                # set tje gradient zero
                optimizer.zero_grad()

                # forward, backward, optimize
                outputs = monk1_network(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print the information of state
                running_loss += loss.item()
                i += 1
                if i % 100 == 99:  # print every 100 batch
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')
        torch.save(monk1_network.state_dict(), 'monk1_model/trained/monk1_network.pkl')
    else:
        monk1_network.load_state_dict(torch.load('monk1_model/trained//monk1_network.pkl'))

    # test the training accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            samples, labels = data
            outputs = monk1_network(samples)
            if outputs[0][0] >= 0.5:
                ans = 1
            else:
                ans = 0

            total += labels.size(0)
            correct += (ans == labels[0]).sum().item()
    print('Accuracy of the network on the all train cases: %.1f %%' % (
                100 * correct / total))

    # test the testing accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            samples, labels = data
            outputs = monk1_network(samples)

            if outputs[0][0] >= 0.5:
                ans = 1
            else:
                ans = 0

            total += labels.size(0)
            correct += (ans == labels[0]).sum().item()
    print('Accuracy of the network on the all test cases: %.1f %%' % (
            100 * correct / total))
    print()

    # calculate or load the hessian inv matrix of the monk1 model
    if not os.path.exists('monk1_model/hessian_inv/monk1_hessian_inv.npy') or re_hessian_inv:
        alpha = 1000000
        H_inv = alpha * np.identity(58)
        for i, data in enumerate(train_loader, 0):
            # get the input
            inputs, labels = data
            labels = labels.unsqueeze(1)

            # set tje gradient zero
            optimizer.zero_grad()

            # forward, backward, optimize
            outputs = monk1_network(inputs)
            loss = criterion(outputs, labels)
            grad_x = torch.autograd.grad(loss, monk1_network.parameters(), create_graph=True)

            X_k = []
            X_k += grad_x[0].view([51]).tolist()
            X_k += grad_x[1].view([3]).tolist()
            X_k += grad_x[2].view([3]).tolist()
            X_k += grad_x[3].view([1]).tolist()
            X_k = np.array(X_k).reshape([58, 1])

            H_inv -= (1 / train_set.samples.shape[0]) * \
                     (np.dot(np.dot(H_inv, X_k), np.dot(np.transpose(X_k), H_inv)) /
                      train_set.samples.shape[0] + np.dot(np.dot(np.transpose(X_k), H_inv), X_k))
            np.save('monk1_model/hessian_inv/monk1_hessian_inv', H_inv, allow_pickle=True, fix_imports=True)
    else:
        H_inv = np.load('monk1_model/hessian_inv/monk1_hessian_inv.npy')

    H_inv = np.abs(H_inv)

    if display_process:
        # find the q that gives the smallest Lq
        count = 0
        Lq = []
        flag = 0
        for i in monk1_network.parameters():
            temp = i.view([np.product(i.size())])
            for j in range(len(temp)):
                this = (1 / 2) * temp[j] * temp[j] / H_inv[count, count]

                Lq.append((this, i, j))
                count += 1

        Lq = sorted(Lq, key=lambda x: x[0])

        # delete weights one by one
        for count, temp in enumerate(Lq):
            print("number of pruned weights: %d" % (count + 1), end='    ')
            temp[1].view([np.product(temp[1].size())])[temp[2]] = 0

            # test the training accuracy
            correct = 0
            total = 0
            with torch.no_grad():
                for data in train_loader:
                    samples, labels = data
                    outputs = monk1_network(samples)
                    if outputs[0][0] >= 0.5:
                        ans = 1
                    else:
                        ans = 0

                    total += labels.size(0)
                    correct += (ans == labels[0]).sum().item()
            print('training accuracy: %.2f%%' % (
                    100 * correct / total), end='    ')

            # test the testing accuracy
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    samples, labels = data
                    outputs = monk1_network(samples)

                    if outputs[0][0] >= 0.5:
                        ans = 1
                    else:
                        ans = 0

                    total += labels.size(0)
                    correct += (ans == labels[0]).sum().item()
            print('testing accuracy: %.2f%%' % (
                    100 * correct / total))
    else:
        # find the q that gives the smallest Lq
        count = 0
        flag = 0
        for i in monk1_network.parameters():
            temp = i.view([np.product(i.size())])
            for j in range(len(temp)):
                this = (1 / 2) * temp[j] * temp[j] / H_inv[count, count]
                if this < threshold:
                    temp[j] = 0
                    flag += 1
                count += 1
        print('pruned %d weights' % flag)

        # test the training accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for data in train_loader:
                samples, labels = data
                outputs = monk1_network(samples)

                if outputs[0][0] >= 0.5:
                    ans = 1
                else:
                    ans = 0

                total += labels.size(0)
                correct += (ans == labels[0]).sum().item()
        print('Accuracy of the network on the all train cases: %.1f %%' % (
                100 * correct / total))

        # test the testing accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                samples, labels = data
                outputs = monk1_network(samples)

                if outputs[0][0] >= 0.5:
                    ans = 1
                else:
                    ans = 0

                total += labels.size(0)
                correct += (ans == labels[0]).sum().item()
        print('Accuracy of the network on the all test cases: %.1f %%' % (
                100 * correct / total))



