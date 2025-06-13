import random

import numpy as np
import torch
import torch.nn as nn
from .utils import block, leaky_block


def sample_label_onehot(n, label_weights):
    # normalize the weights in case they don't sum to 1
    norm_weights = np.array(label_weights) / np.sum(label_weights)

    num_classes = len(norm_weights)
    sample_space = list(range(num_classes))
    labels = np.random.choice(sample_space, size=n, p=norm_weights)

    labels = torch.from_numpy(labels).to(dtype=torch.int64)
    labels_onehot = torch.zeros((n, num_classes))
    labels_onehot[torch.arange(n), labels] = 1

    return labels_onehot, labels


class Generator(nn.Module):
    def __init__(self, bn_on: bool, ln_on: bool, device):
        super().__init__()

        self.bn_on = bn_on
        self.ln_on = ln_on

        self.device = device
        self.pv_dim = 3
        self.lstm_output_dim = 64
        self.noise_dim = 32
        self.layer_num = 1

        self.flow_num_lower = 1
        self.flow_num_upper = 5
        self.flow_len_lower = 10
        self.flow_len_upper = 500

        self.lstm = nn.LSTM(self.noise_dim, self.lstm_output_dim, self.layer_num, batch_first=True)
        if self.ln_on:
            self.ln = nn.LayerNorm(self.lstm_output_dim)

        self.fc = nn.Sequential(
            *block(self.lstm_output_dim, 128, normalize=self.bn_on),
            *block(128, 128, normalize=self.bn_on),
            *block(128, 64, normalize=self.bn_on),
            *block(64, 32, normalize=self.bn_on),
            nn.Linear(32, self.pv_dim),
            nn.Tanh()
        )

    def synthesize(self, noise: torch.Tensor) -> torch.Tensor:
        flow_num, flow_length = noise.size(0), noise.size(1)

        # noise: FLOW_NUMBER * PACKET_DIM
        # length: length of each flow
        # flow = torch.zeros(flow_length, flow_num, self.pv_dim, device=self.device)
        x = noise.to(self.device)
        # h_n = torch.zeros(self.layer_num, flow_num, self.lstm_output_dim, device=self.device)
        # c_n = torch.zeros(self.layer_num, flow_num, self.lstm_output_dim, device=self.device)

        # for i in range(flow_length):
        x, _ = self.lstm(x)
        if self.ln_on:
            x = self.ln(x)

        lstm_output = x.reshape(-1, self.lstm_output_dim)
        fc_output = self.fc(lstm_output)

        # the output of the fully connected layer
        # serves as a single packet vector within the generated flow
        # as well as the input into LSTM at the next step
        # flow[i] = fc_output
        # x = fc_output.unsqueeze(0)
        flow = fc_output.reshape(flow_num, flow_length, self.pv_dim)

        return flow

    def sample_noise(self, n, label_weights) -> tuple[list[torch.Tensor], torch.Tensor]:
        labels_onehot, labels = sample_label_onehot(n, label_weights)

        noises = []
        num_classes = len(label_weights)
        for i in range(n):
            flow_num = random.randint(self.flow_num_lower, self.flow_num_upper)
            length = random.randint(self.flow_len_lower, self.flow_len_upper)

            noise = torch.randn(flow_num, length, self.noise_dim)
            # embed labels into noise vectors
            noise[:, :, :num_classes] = labels_onehot[i]

            noises.append(noise)

        return noises, labels

    def forward(self, noises):
        minibatch = []
        for noise in noises:
            sample = self.synthesize(noise)
            minibatch.append(sample)

        return minibatch


class Generator2(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.noise_dim = 32
        self.feature_dim = 48

        self.fc = nn.Sequential(
            *leaky_block(self.noise_dim, 32),
            *leaky_block(32, 64),
            *leaky_block(64, 128),
            *leaky_block(128, 128),
            *leaky_block(128, 64),
            nn.Linear(64, self.feature_dim)
        )

    def sample_noise(self, n, label_weights):
        labels_onehot, labels = sample_label_onehot(n, label_weights)
        num_classes = len(label_weights)

        noise = torch.randn(n, self.noise_dim)
        noise[:, :num_classes] = labels_onehot
        return noise, labels

    def forward(self, noises):
        output = self.fc(noises)

        return output


if __name__ == '__main__':
    # g = Generator(bn_on=True, ln_on=False, device='cuda').to('cuda')
    g = Generator2(device='cuda').to('cuda')
    sample = g.sample_noise(2, [0.5, 0.2, 0.3, 0, 0, 0])
    print(sample)

