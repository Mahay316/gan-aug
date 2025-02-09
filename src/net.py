import torch.cuda
import torch.nn as nn
from alive_progress import alive_bar
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from model.discriminator import Discriminator
from model.generator import Generator2

from datasets.traffic_dataset import TrafficDataset, get_my_collate
from model.utils import GenLoss, EarlyStopping

from sklearn.metrics import precision_score, recall_score, f1_score
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
gen_label = 7

g = Generator2(device).to(device)
d = Discriminator(bn_on=True, ln_on=False, device=device).to(device)

# print(d.forward(g.generate_batched(4)))

d_criterion = nn.CrossEntropyLoss()
d_optimizer = optim.Adam(d.parameters(), lr=0.004)

g_criterion = GenLoss()
g_optimizer = optim.Adam(g.parameters(), lr=0.001)

scheduler = ExponentialLR(d_optimizer, gamma=0.9)

es_policy = EarlyStopping(patience=15, min_delta=0.01)

root_dir = 'D:/traffic_dataset/span/normalized/'
cd = TrafficDataset(root_dir)
train, val, test = torch.utils.data.random_split(cd, [0.8, 0.1, 0.1])

labels = cd.get_subset_labels(train)
class_weight = 1 / torch.tensor(cd.get_sample_count(), dtype=torch.float) * 100
sample_weights = class_weight[labels]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

training_set = DataLoader(train, batch_size=batch_size, sampler=sampler,
                          collate_fn=get_my_collate())
validating_set = DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=get_my_collate())
test_set = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=get_my_collate())


def train(train_out):
    d.train()

    d_total_loss = 0
    with alive_bar(len(training_set), title=f'Epoch {epoch + 1}') as bar:
        for batch, (trace, label) in enumerate(training_set):
            if len(label) == 0:
                continue
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
            print(f"Batch {batch}, Loss: {d_total_loss}")

    avg_loss = d_total_loss / len(training_set)
    print(f"Discriminator Loss: {avg_loss:.4f}")
    train_out.write(f'{avg_loss}\n')


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

            d.unfreeze_parameters()
            outputs_real = d(samples)
            d_loss_real = d_criterion(outputs_real, labels)

            d_optimizer.zero_grad()
            d_loss_real.backward()
            d_optimizer.step()

            generated_samples = g(batch_size)
            d_loss_gen = torch.zeros(1, device=device)

            if (batch + 1) % 3 == 0:
                d.freeze_parameters()

                generated_labels = torch.tensor([gen_label for _ in range(batch_size)], device=device)
                outputs_gen = d.forward_classifier(generated_samples.detach())
                d_loss_gen = d_criterion(outputs_gen, generated_labels)

                d_optimizer.zero_grad()
                d_loss_gen.backward()
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


def validate(val_out) -> bool:
    d.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        total_loss = 0
        with alive_bar(len(validating_set), title=f'Validating...') as bar:
            for batch, (trace, label) in enumerate(validating_set):
                outputs = d(trace)
                labels = torch.tensor(label, device=device)

                # accumulate validation loss
                val_loss = d_criterion(outputs, labels)
                total_loss += val_loss.item()

                # record accuracy
                predicted = torch.argmax(outputs[:, :-1], dim=1)
                total_correct += (predicted == labels).sum().item()
                total_samples += len(trace)

                bar()

    avg_loss = total_loss / len(test_set)
    acc = total_correct / total_samples * 100

    print(f'Accuracy: {acc: .2f}%')
    print(f"Loss: {avg_loss:.4f}")
    val_out.write(f'{avg_loss}, {acc}\n')
    val_out.flush()

    if avg_loss < es_policy.best_loss:
        torch.save(d.state_dict(), "best_model_d.pth")
        # torch.save(g.state_dict(), "best_model_g.pth")

    return es_policy(avg_loss)


def test(test_out):
    d.load_state_dict(torch.load('model/best_model_d.pth'))
    d.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        with alive_bar(len(test_set), title=f'Testing...') as bar:
            for batch, (trace, label) in enumerate(test_set):
                outputs = d(trace)

                predicted = torch.argmax(outputs[:, :-1], dim=1)
                labels = torch.tensor(label, device=device)

                y_true_list.extend(labels.cpu().numpy())
                y_pred_list.extend(predicted.cpu().numpy())

                bar()

    precision = precision_score(y_true_list, y_pred_list, average=None)
    recall = recall_score(y_true_list, y_pred_list, average=None)
    f1 = f1_score(y_true_list, y_pred_list, average=None)

    test_out.write(f'{precision}\n')
    test_out.write(f'{recall}\n')
    test_out.write(f'{f1}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--train-output', default='train_output.txt')
    parser.add_argument('-v', '--validation-output', default='val_output.txt')
    parser.add_argument('-t', '--test-output', default='test_output.txt')
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    print(args)

    epochs = args.epochs
    train_output = args.train_output
    val_output = args.validation_output
    test_output = args.test_output

    with open(train_output, 'a+') as train_out, open(val_output, 'a+') as val_out:
        for epoch in range(epochs):
            train(train_out)
            validate(val_out)
            scheduler.step()

    with open(test_output, 'a+') as test_out:
        test(test_out)
