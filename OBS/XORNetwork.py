"""
    This code is used to define the xor network,
    train/load the trained xor network,
    calculate/load the hessian matrix of trained xor network
    and show the image of this hessian matrix

    author: Hui Ouyang
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
########################################################################################################################
#   Config: Begin here
########################################################################################################################
# re train the xor network or not if the network is not exit in the file
re_train = False
# number of training iteration
epoch = 100000
# re compute the hessian or not if the hessian is not exit in the file
re_hessian = False
# display the hessian matrix or not
display_hessian = True
# the initial alpha for the hessian matrix (1e-4 to 1e-8 in generally)
alpha = 0.0000001
########################################################################################################################
#   Config: End here
########################################################################################################################


# define the xor network
# structure: 2 - Linear - sigmoid - 2 - Linear - sigmoid - 1
class XORNetwork(nn.Module):

    def __init__(self):
        super(XORNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


if __name__ == '__main__':

    # get a xor network
    xor_network = XORNetwork()

    # get the train data
    input_x = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    input_x = torch.Tensor(input_x)
    y = torch.Tensor(y)

    # train or load the network
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(xor_network.parameters(), lr=0.01, momentum=0.9)
    if not os.path.exists('xor_model/trained/xor_network_100000.pkl') or re_train:
        for _ in range(epoch):
            output_y = xor_network(input_x)
            loss = criterion(output_y, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(xor_network.state_dict(), 'xor_model/trained/xor_network_100000.pkl')
    else:
        xor_network.load_state_dict(torch.load('xor_model/trained//xor_network_100000.pkl'))

    # calculate or load the hessian matrix of xor network
    if not os.path.exists('xor_model/hessian/xor_hessian.npy') or re_hessian:
        H = alpha * np.identity(9)
        for i in range(4):
            output_y = xor_network(input_x[i])
            loss = criterion(output_y, y[i])
            grad_x = torch.autograd.grad(loss, xor_network.parameters(), create_graph=True)
            X_k = []
            X_k += grad_x[2].view([2]).tolist()
            X_k += grad_x[3].view([1]).tolist()
            X_k += grad_x[0].view([4]).tolist()
            X_k += grad_x[1].view([2]).tolist()
            X_k = np.array(X_k).reshape([9, 1])

            H += (1 / 4) * np.dot(X_k, np.transpose(X_k))
            np.save('xor_model/hessian/xor_hessian', H, allow_pickle=True, fix_imports=True)
    else:
        H = np.load('xor_model/hessian/xor_hessian.npy')

    if display_hessian:
        H = np.abs(H)
        # show and shore the image of hessian matrix of xor network
        figure, ax = plt.subplots(1, 1)
        plt.yticks(np.arange(10), ('', 'v1', 'v2', 'v3', 'u11', 'u12', 'u13', 'u21', 'u22', 'u23'))
        ax.matshow(H, cmap=plt.cm.gray)
        plt.xticks(np.arange(9), ('v1', 'v2', 'v3', 'u11', 'u12', 'u13', 'u21', 'u22', 'u23'))
        if not os.path.exists('xor_model/hessian/xor_hessian_img.png'):
            plt.savefig('xor_model/hessian/xor_hessian_img.png')
        figure.show()



