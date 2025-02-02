import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    PACKET_VECTOR_DIM = 5
    FEATURE_DIM = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CLASS_CNT = 5


class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(1, Config.PACKET_VECTOR_DIM, num_layers=3, dropout=0.2, batch_first=True)

    def forward(self, Z):
        return self.lstm(Z)


class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = Config.DEVICE
        self.pv_dim = Config.PACKET_VECTOR_DIM
        self.feature_dim = Config.FEATURE_DIM
        self.class_cnt = Config.CLASS_CNT

        self.conv1 = nn.Conv1d(self.pv_dim, 10, kernel_size=5)
        self.conv2 = nn.Conv1d(10, self.feature_dim // 2, kernel_size=5)
        self.conv3 = nn.Conv1d(self.feature_dim // 2, self.feature_dim, kernel_size=5)

        self.pool1 = nn.MaxPool1d(3, stride=3, padding=1)
        self.pool2 = nn.MaxPool1d(3, stride=3, padding=1)
        self.pool3 = nn.MaxPool1d(3, stride=3, padding=1)

        self.lstm = nn.LSTM(self.feature_dim, self.feature_dim, num_layers=3, dropout=0.2, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(64, self.class_cnt),
            nn.Softmax()
        )

    # flow: [pv1, pv2, ...]
    # pv1: [i, fi, si, ti - t0, ti - t_{i-1}]
    # return: Tensor(type=float32)
    def flow2vector(self, flow: list[list]) -> torch.Tensor:
        flow_tensor = torch.tensor(flow, device=self.device)
        x = flow_tensor.view(-1, self.pv_dim, len(flow))

        padding_thresh = 8

        # reducing input dimension by extracting features from a flow
        # if x.size(2) < padding_thresh:
        #     pad = nn.ConstantPad1d((0, padding_thresh - x.size(2)), value=0)
        #     x = pad(x)
        x = self.pool1(F.leaky_relu(self.conv1(x)))

        # if x.size(2) < padding_thresh:
        #     pad = nn.ConstantPad1d((0, padding_thresh - x.size(2)), value=0)
        #     x = pad(x)
        x = self.pool2(F.leaky_relu(self.conv2(x)))

        # if x.size(2) < padding_thresh:
        #     pad = nn.ConstantPad1d((0, padding_thresh - x.size(2)), value=0)
        #     x = pad(x)
        x = self.pool3(F.leaky_relu(self.conv3(x)))

        # compute flow vector
        # of fixed size Config.feature_size
        cpv = x.view(-1, x.size(2), self.feature_dim)
        _, (h_n, _) = self.lstm(cpv)
        fv = h_n[-1] + h_n[-2]

        return fv

    def fv2type(self, x: torch.Tensor):
        return self.fc(x)

    def forward(self, x):
        cpv = self.conv(x).transpose(-1, -2)
        _, (c, h) = self.lstm(cpv)

        print(h)
        return self.fc(h)


g = Generator().to(Config.DEVICE)
d = Discriminator().to(Config.DEVICE)

flow = [[1, 2, 3.0, 2, 0], [1, 2, 3.0, 2, 0], [1, 2, 3.0, 2, 0], [1, 2, 3.0, 2, 0], [1, 2, 3.0, 2, 0],
            [1, 2, 3.0, 2, 0], [1, 2, 3.0, 2, 0], [1, 2, 3.0, 2, 0]]
data = [
    [flow, 3]
]

batch_size = 64

g_opt = torch.optim.Adam(g.parameters())
d_opt = torch.optim.Adam(d.parameters())


def train():
    for i in range(len(data)):
        x = data[i][0]
        y = data[i][1]


        fv = d.flow2vector(x)

        y_predict = d.fv2type(fv)
        loss = F.cross_entropy(y_predict, y)
        loss.backward()
        loss_show = loss.item()

        if (i + 1) % batch_size == 0:
            pass

if __name__ == '__main__':
    y = torch.tensor(1)
    one = F.one_hot(y, Config.CLASS_CNT)
    print(one)
