import argparse
import math
import os.path

import numpy as np
import torch.cuda
import torch.nn as nn
from alive_progress import alive_bar
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from datasets.traffic_dataset import TrafficDataset, get_my_collate
from model.discriminator import Discriminator
from model.generator import Generator
from model.utils import EarlyStopping, label_smoothing

root_dir = 'D:/traffic_dataset/span/normalized/'

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', default=root_dir, help='path to dataset')

# hyper-parameters
parser.add_argument('--batch-size', type=int, default=64, help='input batch size')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--dis-lr', type=float, default=0.0002, help='learning rate for discriminator')
parser.add_argument('--gen-lr', type=float, default=0.0002, help='learning rate for generator')
parser.add_argument('--scheduler-gamma', type=float, default=0.9, help='decay rate for scheduler')
parser.add_argument('--optimizer-beta1', type=float, default=0.9, help='beta1 for Adam Optimizer')
parser.add_argument('--critic-iter', type=int, default=3, help='discriminator iteration factor')
parser.add_argument('--weighted-loss', action='store_true', help='Balance Cross Entropy weights')
parser.add_argument('--lr-decay', action='store_true', help='Enable Exponential Learning Rate Decay')

# for experiment
parser.add_argument('--drop-class', type=int, default=1, help='minor class label')
parser.add_argument('--drop-rate', type=float, default=0.8, help='minor class drop rate')

# model structure
parser.add_argument('--mode', choices=['gan', 'aug', 'cls'], required=True, help='Enable generator')
parser.add_argument('--disable-generator', action='store_false', dest='generator', help='Disable generator')
parser.add_argument('--enable-bn', type=bool, default=True, help='Enable Batch Normalization')
parser.add_argument('--enable-ln', type=bool, default=False, help='Enable Layer Normalization')

# for training and saving checkpoints
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--label-smoothing', action='store_true', help='enables label smoothing')
parser.add_argument('--gen-pth', default='', help="path to saved parameters of generator")
parser.add_argument('--dis-pth', default='', help="path to saved parameters of discriminator")
parser.add_argument('--out-dir', default='.', help='folder to output model checkpoints')
parser.add_argument('--num_classes', type=int, default=7, help='Number of classes')

opt = parser.parse_args()
print(opt)

os.makedirs(opt.out_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() and opt.cuda else 'cpu'
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

epochs = opt.epoch
batch_size = opt.batch_size
num_classes = opt.num_classes
critic_iter = opt.critic_iter
gen_label = num_classes  # label 0...num_classes - 1 are for real samples, while num_classes is for fake ones
smoothing = opt.label_smoothing

g = Generator(bn_on=True, ln_on=True, device=device).to(device)
d = Discriminator(bn_on=True, ln_on=False, num_classes=num_classes + 1, device=device).to(device)

if opt.dis_pth != '':
    d.load_state_dict(torch.load(opt.dis_pth))

if opt.dis_pth != '':
    g.load_state_dict(torch.load(opt.gen_pth))

d_optimizer = optim.Adam(d.parameters(), lr=opt.dis_lr, betas=(opt.optimizer_beta1, 0.999))
g_optimizer = optim.Adam(g.parameters(), lr=opt.gen_lr, betas=(opt.optimizer_beta1, 0.999))

d_scheduler = ExponentialLR(d_optimizer, gamma=opt.scheduler_gamma)
# g_scheduler = ExponentialLR(g_optimizer, gamma=opt.scheduler_gamma)
es_policy = EarlyStopping(patience=15, min_delta=0.01)

# prepare dataset
cd = TrafficDataset(root_dir)
train_set, val_set, test_set = torch.utils.data.random_split(cd, [0.7, 0.15, 0.15])
# train_set = cd.reduce_subset(train_set, opt.drop_class, opt.drop_rate)

print(np.bincount(cd.get_subset_labels(train_set), minlength=num_classes))

# calculate label weights for gan augmentation
subset_labels = cd.get_subset_labels(train_set)
label_count = np.bincount(subset_labels, minlength=num_classes)
label_weights = np.divide(100, label_count, where=label_count != 0)
print(f'Label Weights: {label_weights}')

criterion = nn.CrossEntropyLoss()
if opt.weighted_loss:
    loss_weights = np.concatenate((label_weights, np.mean(label_weights, keepdims=True)))
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(loss_weights).to(dtype=torch.float32, device=device))

uniform_weights = np.ones(num_classes)

# sample_weights = class_weights[labels]
# sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=get_my_collate())
val_set = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=get_my_collate())
test_set = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=get_my_collate())


def train_classifier(train_out, epoch):
    d.train()

    d_total_loss = 0
    with alive_bar(len(train_set), title=f'Epoch {epoch + 1}') as bar:
        for batch, (trace, label) in enumerate(train_set):
            labels = torch.tensor(label, device=device)

            outputs = d(trace)
            d_loss = criterion(outputs, labels)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            d_total_loss += d_loss.item()

            print(f"Batch {batch}, Running Average Loss: {d_total_loss / (batch + 1):.4f}")

            # update the progress bar
            bar()

    avg_loss = d_total_loss / len(train_set)
    # print(f"Discriminator Loss: {avg_loss:.4f}")
    train_out.write(f'{avg_loss}\n')
    train_out.flush()


def train_gan(train_out, epoch):
    d.train()
    g.train()

    d_total_loss_real = 0
    d_total_loss_fake = 0
    g_total_loss = 0
    with alive_bar(len(train_set), title=f'Epoch {epoch + 1}') as bar:
        for batch, (trace, label) in enumerate(train_set):
            # ================ TRAIN DISCRIMINATOR ===============
            # real samples extracted from dataset
            real_samples = trace
            real_labels = torch.tensor(label)
            if smoothing:
                real_labels = label_smoothing(real_labels, num_classes + 1)
            real_labels = real_labels.to(device=device)

            real_pred_labels = d(real_samples)
            d_loss_real = criterion(real_pred_labels, real_labels)

            # fake samples synthesized by the generator
            # for optimal balance when considered synthesized as a standalone class
            fake_count = math.ceil(batch_size / num_classes)
            fake_labels = torch.tensor([gen_label for _ in range(fake_count)])
            if True:
                fake_labels = label_smoothing(fake_labels, num_classes + 1, smoothing=0.1)
            fake_labels = fake_labels.to(device=device)

            fake_noises, _ = g.sample_noise(fake_count, uniform_weights)
            fake_samples = [t.detach() for t in g(fake_noises)]

            fake_pred_labels = d(fake_samples)
            d_loss_fake = criterion(fake_pred_labels, fake_labels)

            d_loss = d_loss_real + d_loss_fake

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            # ================ TRAIN DISCRIMINATOR ===============

            # ================== TRAIN GENERATOR =================
            # train discriminator for `critic_iter` times before training generator
            if (batch + 1) % critic_iter == 0:
                gen_noises, gen_labels = g.sample_noise(batch_size, label_weights)
                if smoothing:
                    gen_labels = label_smoothing(gen_labels, num_classes + 1)
                gen_labels = gen_labels.to(device=device)

                gen_samples = g(gen_noises)
                gen_pred_labels = d(gen_samples)
                g_loss = criterion(gen_pred_labels, gen_labels)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                g_total_loss += g_loss.item()
            # ================== TRAIN GENERATOR =================

            d_total_loss_real += d_loss_real.item()
            d_total_loss_fake += d_loss_fake.item()
            bar()
            print(
                f'Batch {batch}, Running Average Loss:',
                f'Dr = {d_total_loss_real / (batch + 1):.4f}',
                f'Df = {d_total_loss_fake / (batch + 1):.4f}',
                f'G = {g_total_loss / ((batch + 1) / critic_iter):.4f}'
            )

    # print(f"Epoch [{epoch + 1}/{epochs}]")
    # print(f"Discriminator Loss: {d_total_loss / len(train_set):.4f}")
    # print(f"Generator Loss: {g_total_loss / len(train_set):.4f}")
    train_out.write(f'{d_total_loss_real / len(train_set):.4f}, '
                    f'{d_total_loss_fake / len(train_set):.4f}, '
                    f'{g_total_loss / len(train_set) * critic_iter:.4f}\n')
    train_out.flush()


def augment_discriminator(train_out, epoch):
    d.train()

    d_total_loss_real = 0
    d_total_loss_fake = 0
    with alive_bar(len(train_set), title=f'Epoch {epoch + 1}') as bar:
        for batch, (trace, label) in enumerate(train_set):
            # ================ TRAIN DISCRIMINATOR ===============
            # real samples extracted from dataset
            real_samples = trace
            real_labels = torch.tensor(label).to(device=device)

            real_pred_labels = d(real_samples)
            d_loss_real = criterion(real_pred_labels, real_labels)

            # fake samples synthesized by the generator
            with torch.no_grad():
                fake_noises, fake_labels = g.sample_noise(batch_size, label_weights)
                if True:
                    fake_labels = label_smoothing(fake_labels, num_classes + 1, smoothing=0.1)
                fake_labels = fake_labels.to(device=device)
                fake_samples = g(fake_noises)

            fake_pred_labels = d([t.detach() for t in fake_samples])
            d_loss_fake = criterion(fake_pred_labels, fake_labels)

            d_loss = d_loss_real + d_loss_fake

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            d_total_loss_real += d_loss_real.item()
            d_total_loss_fake += d_loss_fake.item()
            print(f"Batch {batch}, Running Average Loss: Dr = {d_total_loss_real / (batch + 1):.4f}, "
                  f"Df = {d_total_loss_fake / (batch + 1):.4f}")

            # update the progress bar
            bar()

        train_out.write(f'{d_total_loss_real / (batch + 1):.4f}, '
                        f'{d_total_loss_fake / (batch + 1):.4f}\n')
        train_out.flush()


def validate(val_out, epoch) -> bool:
    d.eval()
    y_true_list, y_pred_list = [], []
    total_loss = 0
    with torch.no_grad():
        with alive_bar(len(val_set), title=f'Validating...') as bar:
            for batch, (trace, label) in enumerate(val_set):
                outputs = d(trace)

                # accumulate validation loss
                labels = torch.tensor(label, device=device)
                val_loss = criterion(outputs, labels)
                total_loss += val_loss.item()

                pred_label = torch.argmax(outputs[:, :-1], dim=1)

                y_true_list.extend(label)
                y_pred_list.extend(pred_label.cpu().numpy())

                bar()

    avg_loss = total_loss / len(val_set)
    accuracy = accuracy_score(y_true_list, y_pred_list)
    precision = precision_score(y_true_list, y_pred_list, average=None, zero_division=0.0)
    recall = recall_score(y_true_list, y_pred_list, average=None, zero_division=0.0)
    f1 = f1_score(y_true_list, y_pred_list, average=None, zero_division=0.0)

    val_out.write(f'Epoch: {epoch + 1}\n')
    val_out.write(f'Loss: {avg_loss}\n')
    val_out.write(f'Accuracy: {accuracy}\n')
    val_out.write(f'Precision: {precision.tolist()}\n')
    val_out.write(f'Recall: {recall.tolist()}\n')
    val_out.write(f'F1: {f1.tolist()}\n')
    val_out.write('\n')
    val_out.flush()

    if avg_loss < es_policy.best_loss:
        torch.save(d.state_dict(), os.path.join(opt.out_dir, 'd_best.pth'))

    return es_policy(avg_loss)


def test(test_out, param_file):
    d.load_state_dict(torch.load(param_file))
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

    precision = precision_score(y_true_list, y_pred_list, average=None, zero_division=0.0)
    recall = recall_score(y_true_list, y_pred_list, average=None, zero_division=0.0)
    f1 = f1_score(y_true_list, y_pred_list, average=None, zero_division=0.0)

    test_out.write(f'{precision}\n')
    test_out.write(f'{recall}\n')
    test_out.write(f'{f1}\n')
    test_out.flush()

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Generate the confusion matrix
    cm = confusion_matrix(y_true_list, y_pred_list, labels=range(num_classes))

    # Visualize the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Class {i}" for i in range(num_classes)])
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


train_output = os.path.join(opt.out_dir, 'train.txt')
val_output = os.path.join(opt.out_dir, 'val.txt')
test_output = os.path.join(opt.out_dir, 'test.txt')

with open(train_output, 'a+') as train_out, open(val_output, 'a+') as val_out:
    for epoch in range(epochs):
        if opt.mode == 'gan':
            train_gan(train_out, epoch)
        elif opt.mode == 'aug':
            augment_discriminator(train_out, epoch)
        elif opt.mode == 'cls':
            train_classifier(train_out, epoch)
        else:
            raise RuntimeError('Invalid choice of mode!')

        # save checkpoints
        torch.save(d.state_dict(), os.path.join(opt.out_dir, f'dis_model_{epoch}.pth'))
        if opt.generator:
            torch.save(g.state_dict(), os.path.join(opt.out_dir, f'gen_model_{epoch}.pth'))

        validate(val_out, epoch)
        if opt.lr_decay:
            d_scheduler.step()

        # if opt.generator:
        # g_scheduler.step()

# with open(test_output, 'a+') as test_out:
#     test(test_out, os.path.join(opt.out_dir, 'dis_model_28.pth'))
