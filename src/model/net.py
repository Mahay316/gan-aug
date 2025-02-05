import torch.cuda
import torch.nn as nn
from alive_progress import alive_bar
from torch import optim
from torch.utils.data import DataLoader

from discriminator import Discriminator
from generator import Generator2
from src.datasets.traffic_dataset import TrafficDataset, my_collate_fn
from utils import GenLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
gen_label = 7

g = Generator2(device).to(device)
d = Discriminator(device).to(device)

# print(d.forward(g.generate_batched(4)))

d_criterion = nn.CrossEntropyLoss()
d_optimizer = optim.Adam(d.parameters(), lr=0.005)

g_criterion = GenLoss()
g_optimizer = optim.Adam(g.parameters(), lr=0.005)

root_dir = 'D:/traffic_dataset/normalized_dataset'
cd = TrafficDataset(root_dir)

train, val, test = torch.utils.data.random_split(cd, [0.7, 0.2, 0.1])
training_set = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
validating_set = DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)


def train():
    d.train()
    g.train()

    d_total_loss = 0
    g_total_loss = 0
    with alive_bar(len(training_set), title=f'Epoch {epoch + 1}') as bar:
        for batch, (trace, label) in enumerate(training_set):
            # generated_samples = g(batch_size)

            # ================ TRAIN DISCRIMINATOR ===============
            # merge generated samples with real samples
            # fix generator's parameters when training discriminator
            # samples = [item.detach() for item in generated_samples] + trace

            # create labels
            # generated_labels = [gen_label for _ in range(batch_size)]
            # labels = torch.tensor(generated_labels + label, device=device)

            samples = trace
            labels = torch.tensor(label, device=device)

            outputs = d(samples)
            d_loss = d_criterion(outputs, labels)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            # ================ TRAIN DISCRIMINATOR ===============

            # ================== TRAIN GENERATOR =================
            # outputs = d(generated_samples)
            # g_loss = g_criterion(outputs)
            #
            # g_optimizer.zero_grad()
            # g_loss.backward()
            # g_optimizer.step()
            # ================== TRAIN GENERATOR =================

            d_total_loss += d_loss.item()
            # g_total_loss += g_loss.item()
            bar()
            print(f"Batch {batch}, Loss: {d_total_loss} {g_total_loss}")

    # print(f"Epoch [{epoch + 1}/{epochs}]")
    print(f"Discriminator Loss: {d_total_loss / len(training_set):.4f}")
    print(f"Generator Loss: {g_total_loss / len(training_set):.4f}")


def train2():
    d.train()
    g.train()

    d_total_loss = 0
    g_total_loss = 0
    with alive_bar(len(training_set), title=f'Epoch {epoch + 1}') as bar:
        for batch, (trace, label) in enumerate(training_set):
            # ================ TRAIN DISCRIMINATOR ===============
            samples = trace
            labels = torch.tensor(label, device=device)



            d.freeze_parameters()
            generated_samples = g(batch_size)
            generated_labels = torch.tensor([gen_label for _ in range(batch_size)], device=device)
            outputs_gen = d.forward_classifier(generated_samples.detach())
            d_loss_gen = d_criterion(outputs_gen, generated_labels)

            d_optimizer.zero_grad()
            d_loss_gen.backward()
            d_optimizer.step()

            d.unfreeze_parameters()
            outputs_real = d(samples)
            d_loss_real = d_criterion(outputs_real, labels)

            d_optimizer.zero_grad()
            d_loss_real.backward()
            d_optimizer.step()
            # ================ TRAIN DISCRIMINATOR ===============

            # ================== TRAIN GENERATOR =================
            outputs = d.forward_classifier(generated_samples)
            g_loss = g_criterion(outputs)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            # ================== TRAIN GENERATOR =================

            d_total_loss += d_loss_gen.item() + d_loss_real.item()
            g_total_loss += g_loss.item()
            bar()
            print(f"Batch {batch}, Loss: {d_total_loss} {g_total_loss}")

    # print(f"Epoch [{epoch + 1}/{epochs}]")
    print(f"Discriminator Loss: {d_total_loss / len(training_set):.4f}")
    print(f"Generator Loss: {g_total_loss / len(training_set):.4f}")


def validate():
    d.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        with alive_bar(len(validating_set), title=f'Validating...') as bar:
            for batch, (trace, label) in enumerate(validating_set):
                outputs = d(trace)
                _, predicted = torch.max(outputs.data, 1)

                labels = torch.tensor(label, device=device)
                total_correct += (predicted == labels).sum().item()
                total_samples += len(trace)

                bar()
                # d_loss = d_criterion(outputs, labels)
        print(f'Accuracy: {total_correct / total_samples * 100: .2f}%')


epochs = 20
for epoch in range(epochs):
    train2()
    validate()
