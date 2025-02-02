import torch.cuda
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from src.datasets.traffic_dataset import TrafficDataset, my_collate_fn

from generator import Generator
from discriminator import Discriminator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
gen_label = 6

g = Generator('cpu')
d = Discriminator(device).to(device)

# print(d.forward(g.generate_batched(4)))

d_criterion = nn.CrossEntropyLoss()
d_optimizer = optim.Adam(d.parameters(), lr=0.001)

g_criterion = nn.CrossEntropyLoss
g_optimizer = optim.Adam(g.parameters(), lr=0.001)

root_dir = 'D:/BaiduNetdiskDownload/Tor Traffic/'
cd = TrafficDataset(root_dir)

train, val, test = torch.utils.data.random_split(cd, [0.8, 0.1, 0.1])
dl = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

epochs = 20
for epoch in range(epochs):
    d.train()
    total_loss = 0

    for batch, (trace, label) in enumerate(dl):
        # merge generated samples with real samples
        # generated_samples = g.generate_batched(batch_size)
        # samples = generated_samples + trace

        # create labels
        # generated_labels = [gen_label for _ in range(batch_size)]
        # labels = generated_labels + label
        labels = torch.tensor(label, device=device)

        outputs = d.forward(trace)
        d_loss = d_criterion(outputs, labels)

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        total_loss += d_loss.item()

        print(f"Batch {batch}, progress: {(batch + 1) / len(dl) * 100:.2f}%")

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dl):.4f}")
