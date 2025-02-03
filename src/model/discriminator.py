import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from generator import Generator


class Discriminator(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.pv_dim = 3
        self.feature_dim = 10
        self.class_cnt = 7

        self.device = device

        # Convolutional layers for packet vector compression
        self.conv1 = nn.Conv1d(in_channels=self.pv_dim, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool1d(3, stride=3, padding=1)
        self.conv2 = nn.Conv1d(10, self.feature_dim // 2, kernel_size=5)
        self.pool2 = nn.MaxPool1d(3, stride=3, padding=1)
        self.conv3 = nn.Conv1d(self.feature_dim // 2, self.feature_dim, kernel_size=5)
        self.pool3 = nn.MaxPool1d(3, stride=3, padding=1)

        self.flowModule = nn.LSTM(self.feature_dim, self.feature_dim, num_layers=3, dropout=0.2, batch_first=True)
        self.traceModule = nn.LSTM(self.feature_dim, self.feature_dim, num_layers=3, dropout=0.2, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(64, self.class_cnt)
        )

    # seq: a single piece of flow of packets
    def compress_packet_vector(self, trace_tensor):
        # seq_tensor TRACE_LENGTH * FLOW_LENGTH * PACKET_DIM
        seq_tensor = trace_tensor.to(self.device).swapaxes(1, 2)

        # pad the sequence of packets if they are not long enough
        # 7 Conv[7 - (kernel - 1)] -> 3 MaxPool
        minimum_input_length = 7

        if seq_tensor.size(2) < minimum_input_length:
            seq_tensor = F.pad(seq_tensor, (0, minimum_input_length - seq_tensor.size(2)), "constant", 0)
        seq_tensor = self.pool1(F.leaky_relu(self.conv1(seq_tensor)))

        if seq_tensor.size(2) < minimum_input_length:
            seq_tensor = F.pad(seq_tensor, (0, minimum_input_length - seq_tensor.size(2)), "constant", 0)
        seq_tensor = self.pool2(F.leaky_relu(self.conv2(seq_tensor)))

        if seq_tensor.size(2) < minimum_input_length:
            seq_tensor = F.pad(seq_tensor, (0, minimum_input_length - seq_tensor.size(2)), "constant", 0)
        seq_tensor = self.pool3(F.leaky_relu(self.conv3(seq_tensor)))

        return seq_tensor.swapaxes(1, 2)

    def compute_flow_vector(self, trace_tensor):
        out, (h_n, c_n) = self.flowModule(trace_tensor)

        # only the final hidden state of the last two layers of LSTM are considered
        x = h_n[-1, :] + h_n[-2, :]

        return x

    def compute_trace_vector(self, flow_vectors):
        out, (h_n, c_n) = self.traceModule(flow_vectors)

        # only the final hidden state of the last two layers of LSTM are considered
        x = h_n[-1, :] + h_n[-2, :]

        return x

    # data_in is batched !Python list!
    # because traces are of different shapes
    # shape = batch_size * trace_shape
    def forward(self, data_in):
        trace_vector_list = []
        for trace in data_in:
            cpv = self.compress_packet_vector(trace)
            fv = self.compute_flow_vector(cpv)
            tv = self.compute_trace_vector(fv)

            trace_vector_list.append(tv)

        trace_batch = torch.stack(trace_vector_list)
        return self.classifier(trace_batch)

    def my_train(self, dataloader, batch_size, optimizer, epoch_num):
        self.train()

        for epoch in range(epoch_num):
            for i, (X, Y) in enumerate(dataloader):
                predict = self.forward(X)
                y_pred = predict.argmax(dim=1, keepdim=True)

                y_ground = torch.tensor(Y)
                loss = F.cross_entropy(y_pred, y_ground)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


if __name__ == '__main__':
    g = Generator('cuda').to('cuda')
    d = Discriminator('cuda').to('cuda')

    t1 = time.time()
    gen = g(32)
    t2 = time.time()
    d([x.detach() for x in gen])
    d(gen)
    t3 = time.time()
    print(t2 - t1)
    print(t3 - t2)
