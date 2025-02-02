import random

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.pv_dim = 3
        self.lstm_output_dim = 64
        self.noise_dim = 3
        self.layer_num = 3

        self.flow_num_lower = 1
        self.flow_num_upper = 5
        self.flow_len_lower = 10
        self.flow_len_upper = 200

        self.lstm = nn.LSTM(self.noise_dim, self.lstm_output_dim, self.layer_num)
        self.fc = nn.Sequential(
            nn.Linear(self.lstm_output_dim, 32),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(16, self.pv_dim),
            nn.Dropout(),
            nn.LeakyReLU()
        )

    def forward(self, noise, length) -> list[torch.Tensor]:
        # noise: FLOW_NUMBER * PACKET_DIM
        # length: length of each flow
        trace = []

        for i in range(len(length)):
            flow_length = length[i]
            flow = torch.zeros(flow_length, self.pv_dim).to(self.device)

            x = noise[i].unsqueeze(0).to(self.device)
            h_n = torch.zeros(self.layer_num, self.lstm_output_dim, device=self.device)
            c_n = torch.zeros(self.layer_num, self.lstm_output_dim, device=self.device)

            for j in range(flow_length):
                _, (h_n, c_n) = self.lstm(x, (h_n, c_n))

                lstm_output = h_n[-1]
                fc_output = self.fc(lstm_output)

                # the output of the fully connected layer
                # serves as a single packet vector within the generated flow
                # as well as the input into LSTM at the next step
                flow[j] = fc_output
                x = fc_output.unsqueeze(0)

            trace.append(flow)

        return trace

    def generate_batched(self, batch_size):
        minibatch = []
        for i in range(batch_size):
            # randomly sampled noise to be fed into the generator
            flow_num = random.randint(self.flow_num_lower, self.flow_num_upper)
            length = random.randint(self.flow_len_lower, self.flow_len_upper)

            noise = torch.randn(flow_num, self.noise_dim)

            sample = self.forward(noise, length)
            minibatch.append(sample)

        return minibatch
