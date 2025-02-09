from utils import block
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, bn_on: bool, ln_on: bool, device):
        """
        :param bn_on: whether enable batch normalization or not
        :param ln_on: whether enable layer normalization or not
        :param device: which device to put this model on
        """

        super().__init__()

        self.bn_on = bn_on
        self.ln_on = ln_on

        self.pv_dim = 3
        self.feature_dim = 16
        self.class_cnt = 8

        self.device = device

        self.flow_feature_dim = 32
        self.trace_feature_dim = 48

        # Convolutional layers for packet vector compression
        self.conv1 = nn.Conv1d(in_channels=self.pv_dim, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool1d(3, stride=3, padding=1)
        self.conv2 = nn.Conv1d(10, self.feature_dim // 2, kernel_size=5)
        self.pool2 = nn.MaxPool1d(3, stride=3, padding=1)
        self.conv3 = nn.Conv1d(self.feature_dim // 2, self.feature_dim, kernel_size=5)
        self.pool3 = nn.MaxPool1d(3, stride=3, padding=1)

        self.flowModule = nn.LSTM(self.feature_dim, self.flow_feature_dim, num_layers=1, batch_first=True)
        self.flow_ln = nn.LayerNorm(self.flow_feature_dim)

        self.traceModule = nn.LSTM(self.flow_feature_dim, self.trace_feature_dim, num_layers=1, batch_first=True)
        self.trace_ln = nn.LayerNorm(self.trace_feature_dim)

        self.classifier = nn.Sequential(
            *block(self.trace_feature_dim, 64, normalize=self.bn_on),
            *block(64, 128, normalize=self.bn_on),
            *block(128, 256, normalize=self.bn_on),
            *block(256, 128, normalize=self.bn_on),
            *block(128, 64, normalize=self.bn_on),
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
        if self.ln_on:
            x = self.flow_ln(x)

        return x

    def compute_trace_vector(self, flow_vectors):
        out, (h_n, c_n) = self.traceModule(flow_vectors)

        # only the final hidden state of the last two layers of LSTM are considered
        x = h_n[-1, :]  # + h_n[-2, :]
        if self.ln_on:
            x = self.trace_ln(x)

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
