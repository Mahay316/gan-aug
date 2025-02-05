import time
from utils import block
import torch
import torch.nn as nn
import torch.nn.functional as F

from generator import Generator


class Discriminator(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.pv_dim = 3
        self.feature_dim = 10
        self.class_cnt = 8

        self.device = device

        # Convolutional layers for packet vector compression
        self.conv1 = nn.Conv1d(in_channels=self.pv_dim, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool1d(3, stride=3, padding=1)
        self.conv2 = nn.Conv1d(10, self.feature_dim // 2, kernel_size=5)
        self.pool2 = nn.MaxPool1d(3, stride=3, padding=1)
        self.conv3 = nn.Conv1d(self.feature_dim // 2, self.feature_dim, kernel_size=5)
        self.pool3 = nn.MaxPool1d(3, stride=3, padding=1)

        self.flowModule = nn.LSTM(self.feature_dim, self.feature_dim, num_layers=1, batch_first=True)
        self.traceModule = nn.LSTM(self.feature_dim, self.feature_dim, num_layers=1, batch_first=True)

        self.classifier = nn.Sequential(
            *block(self.feature_dim, 64),
            *block(64, 128),
            *block(128, 256),
            *block(256, 128),
            *block(128, 64),
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
        x = h_n[-1, :]  # + h_n[-2, :]

        return x

    def compute_trace_vector(self, flow_vectors):
        out, (h_n, c_n) = self.traceModule(flow_vectors)

        # only the final hidden state of the last two layers of LSTM are considered
        x = h_n[-1, :]  # + h_n[-2, :]

        return x

    def forward_classifier(self, trace_batch):
        return self.classifier(trace_batch)

    def freeze_parameters(self, exclude='classifier'):
        for name, param in self.named_parameters():
            if exclude not in name:
                param.requires_grad = False

    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, data_in):
        trace_vector_list = []
        for trace in data_in:
            cpv = self.compress_packet_vector(trace)
            fv = self.compute_flow_vector(cpv)
            tv = self.compute_trace_vector(fv)

            trace_vector_list.append(tv)

        trace_batch = torch.stack(trace_vector_list)
        return self.forward_classifier(trace_batch)


if __name__ == '__main__':
    d = Discriminator('cpu')
    d.freeze_parameters()
    for name, param in d.named_parameters():
        print(f'{name}: {param.requires_grad}')

    d.unfreeze_parameters()
    for name, param in d.named_parameters():
        print(f'{name}: {param.requires_grad}')
