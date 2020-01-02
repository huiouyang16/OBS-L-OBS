import numpy as np


class PruneUtil:
    def __init__(self):
        pass

    def edge_cut(self, w_layer, b_layer, hessian_inverse, cut_ratio):
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
            sensitivity = np.hstack(
                (sensitivity, 0.5 * (np.hstack((w_layer.T[i], b_layer[i])) ** 2) / np.diag(hessian_inverse)))
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
                    delta = (-w_layer[y_index][x_index] / hessian_inverse[y_index][y_index]) * hessian_inverse.T[
                        y_index]
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