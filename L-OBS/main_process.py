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

    def produce_layer_input(self, x):
        in1 = self.input_layer(x)
        in2 = torch.relu(in1)
        hidden1 = self.hidden_layer(in2)
        hidden2 = torch.relu(hidden1)
        return x, in2, hidden2


n = N()
n.load_state_dict(torch.load('params_lenet_300_100.pkl'))
para = list(n.parameters())
w_input_layer = para[0].detach().numpy()
b_input_layer = para[1].detach().numpy()
w_hidden_layer = para[2].detach().numpy()
b_hidden_layer = para[3].detach().numpy()
w_output_layer = para[4].detach().numpy()
b_output_layer = para[5].detach().numpy()


inputs = []
for i in range(100):
    inputs.append([])
i = 1
for images, _ in testloader:
    # print(i)
    images = images.reshape(-1, 28 * 28)
    a, b, c = n.produce_layer_input(images)
    a, b, c = a.numpy(), b.detach().numpy(), c.detach().numpy()
    # print(a.shape, b.shape, c.shape)
    inputs[i - 1].append(a)
    inputs[i - 1].append(b)
    inputs[i - 1].append(c)
    # print(len(images))
    # print('------')
    i += 1
print('Layer inputs produced')


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


def generate_hessian_inverse(layer):
    l = -1
    num_layer = 0
    if layer == 'input_layer':
        l = 0
        num_layer = w_input_layer.shape[1]
    elif layer == 'hidden_layer':
        l = 1
        num_layer = w_hidden_layer.shape[1]
    elif layer == 'output_layer':
        l = 2
        num_layer = w_output_layer.shape[1]

    # initialize hessian inverse
    # i.e. PSI.inv for l-1
    hessian_inverse = 1000000 * np.eye(num_layer + 1)

    for index, input_batch in enumerate(inputs):
        input_batch_item = input_batch[l]
        if index == 0:
            batch_size = input_batch_item.shape[0]
            n = input_batch_item.shape[0] * 100
        for i in range(batch_size):
            vector_y = np.vstack((np.array([input_batch_item[i]]).T, np.array([[1.0]])))
            # numerator = PSI.inv dot y dot y.T dot PSI.inv
            numerator = np.dot(np.dot(hessian_inverse, vector_y), np.dot(vector_y.T, hessian_inverse))
            # denominator = n + y.T dot PSI.inv dot y
            denominator = n + np.dot(np.dot(vector_y.T, hessian_inverse), vector_y)
            # next PSI.inv = PSI.inv - numerator / denominator
            hessian_inverse = hessian_inverse - numerator * (1.00 / denominator)

    print('Hessian inverse generated')
    return hessian_inverse


hessian_inverse_input_layer = generate_hessian_inverse('input_layer')
hessian_inverse_hidden_layer = generate_hessian_inverse('hidden_layer')
hessian_inverse_output_layer = generate_hessian_inverse('output_layer')


def edge_cut(w_layer, b_layer, hessian_inverse, cut_ratio):
    w_layer = w_layer.T

    num_hidden_1 = w_layer.shape[0]
    num_hidden_2 = w_layer.shape[1]
    num_weights = num_hidden_1 * num_hidden_2
    max_prune_num = int(num_weights * cut_ratio)
    print('To prune:', max_prune_num)

    # initialize sensitivity
    sensitivity = np.array([])

    w_gate = np.ones([num_hidden_1, num_hidden_2])
    b_gate = np.ones([num_hidden_2])

    for i in range(num_hidden_2):
        # every recursion, do for a 28*28 sample
        # put all of them in a line
        sensitivity = np.hstack((sensitivity, 0.5 * (np.hstack((w_layer.T[i], b_layer[i])) ** 2) / np.diag(hessian_inverse)))
    # sort all the sensitivity
    sorted_index = np.argsort(sensitivity)

    print('Start pruning')
    count = 0
    for i in range(num_weights):
        prune_index = sorted_index[i]
        # which layer: l
        # i.e. index out of all the samples
        x_index = prune_index // (num_hidden_1 + 1)
        # layer location: q
        # i.e. index out of one sample
        y_index = prune_index % (num_hidden_1 + 1)

        # bias layer
        if y_index == num_hidden_1:
            if b_gate[x_index] == 1:
                delta = (-b_layer[x_index] / hessian_inverse[y_index][y_index]) * hessian_inverse.T[y_index]
                b_gate[x_index] = 0
                # update
                w_layer.T[x_index] = w_layer.T[x_index] + delta[0: -1]
                b_layer[x_index] = b_layer[x_index] + delta[-1]
                count += 1
        # not bias layer
        else:
            if w_gate[y_index][x_index] == 1:
                delta = (-w_layer[y_index][x_index] / hessian_inverse[y_index][y_index]) * hessian_inverse.T[y_index]
                w_gate[y_index][x_index] = 0
                # update
                w_layer.T[x_index] = w_layer.T[x_index] + delta[0: -1]
                b_layer[x_index] = b_layer[x_index] + delta[-1]
                count += 1

        w_layer = w_layer * w_gate
        b_layer = b_layer * b_gate
        # print('Pruned: ', count, ' / ', max_prune_num)

        if count == max_prune_num:
            break

    print('Finish pruning')

    w_layer = w_layer.T

    return w_layer, b_layer


w_input_layer_cut, b_input_layer_cut = edge_cut(w_input_layer, b_input_layer, hessian_inverse_input_layer, 0.933)
w_hidden_layer_cut, b_hidden_layer_cut = edge_cut(w_hidden_layer, b_hidden_layer, hessian_inverse_hidden_layer, 0.8)
w_output_layer_cut,b_output_layer_cut = edge_cut(w_output_layer, b_output_layer, hessian_inverse_output_layer, 0.35)

w_input_layer_cut = torch.tensor(w_input_layer_cut)
b_input_layer_cut = torch.tensor(b_input_layer_cut)
w_hidden_layer_cut = torch.tensor(w_hidden_layer_cut)
b_hidden_layer_cut = torch.tensor(b_hidden_layer_cut)
w_output_layer_cut = torch.tensor(w_output_layer_cut)
b_output_layer_cut = torch.tensor(b_output_layer_cut)

p = torch.load('params_lenet_300_100.pkl')
p['input_layer.weight'] = w_input_layer_cut
p['input_layer.bias'] = b_input_layer_cut
p['hidden_layer.weight'] = w_hidden_layer_cut
p['hidden_layer.bias'] = b_hidden_layer_cut
p['output_layer.weight'] = w_output_layer_cut
p['output_layer.bias'] = b_output_layer_cut
torch.save(p, 'params_lenet_300_100_pruned.pkl')
