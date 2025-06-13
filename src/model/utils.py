import math

import torch
import torch.nn as nn


def leaky_block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.LeakyReLU())
    return layers


def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.ReLU())
    return layers


def label_smoothing(labels, num_classes, smoothing=0.1):
    batch_size = labels.size(0)
    smoothed_labels = torch.full((batch_size, num_classes), smoothing / (num_classes - 1))
    smoothed_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)

    return smoothed_labels


class GenLoss(nn.Module):
    def __init__(self, eps=1e-10):
        super(GenLoss, self).__init__()
        self.soft = nn.Softmax(dim=1)
        self.eps = eps

    def forward(self, x):
        x = self.soft(x)
        x = -torch.log(1 - x[:, -1] + self.eps)

        # not_inf_idx = ~torch.isinf(x)

        return torch.mean(x)


class EarlyStopping:
    def __init__(self, patience, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = math.inf
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print('Early Stopping triggered')
                return True
        return False


if __name__ == '__main__':
    labels = torch.randint(0, 6, (32,))
    b = label_smoothing(labels, 6)

    print(b)
