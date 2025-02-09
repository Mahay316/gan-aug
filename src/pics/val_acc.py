import matplotlib.pyplot as plt
import numpy as np
import os

label_to_filename = {
    'Baseline': 'val_acc_nn.csv',
    'BatchNorm': 'val_acc_bn.csv',
    'LayerNorm': 'val_acc_ln.csv',
    'BatchNorm + LayerNorm': 'val_acc.csv',
}


def find_convergence_epoch(accuracies, window=10, threshold=1):
    for epoch in range(len(accuracies) - window + 1):
        window_data = accuracies[epoch:epoch + window]
        if np.ptp(window_data) <= threshold:
            return epoch + window
    return len(accuracies) - 1


steps_per_epoch = 106
data_dir = './val_acc/'

fig, ax = plt.subplots(dpi=300)
ax.set_xlabel('Training steps')
ax.set_ylabel('Validation Accuracy/%')

for label, filename in label_to_filename.items():
    with open(os.path.join(data_dir, filename), 'r') as f:
        val_acc = np.array([float(line.split(', ')[1]) for line in f])

    training_step = np.arange(steps_per_epoch, (len(val_acc) + 1) * steps_per_epoch, step=steps_per_epoch)
    ax.plot(training_step, val_acc, linewidth=1, label=label)

    conv_epoch = find_convergence_epoch(val_acc)
    conv_acc = val_acc[conv_epoch]
    conv_step = conv_epoch * steps_per_epoch

    # employ the same color as the line for the asterisk marker and dashed line
    color = plt.gca().lines[-1].get_color()
    plt.scatter(conv_step, conv_acc, marker='*', s=80, color=color, zorder=5)
    plt.axvline(conv_step, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    plt.text(conv_step + 12 * steps_per_epoch, conv_acc + 2, f'Step {conv_step}',
             color=color, ha='center', va='top')

ax.set_xlim(0)
ax.set_ylim(40)
ax.grid(linestyle=':')
ax.legend(loc='best')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.show()
