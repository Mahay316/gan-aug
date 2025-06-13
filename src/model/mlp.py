import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=12, output_dim=7):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    model = MLP(input_dim=12, output_dim=7)
    print(model)
    x = torch.randn(5, 12)
    output = model(x)
    print("\nInput shape:", x.shape)
    print("Output shape:", output.shape)
