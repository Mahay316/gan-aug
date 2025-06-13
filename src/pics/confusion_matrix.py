import torch
import numpy as np
import matplotlib.pyplot as plt

# Example: True and predicted labels
true_labels = torch.tensor([0, 1, 2, 2, 0, 1, 2, 1, 0])  # Ground truth
predicted_labels = torch.tensor([0, 2, 2, 2, 0, 1, 1, 1, 0])  # Model predictions

# Number of classes
num_classes = 6

# Compute confusion matrix
# conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)
# for t, p in zip(true_labels, predicted_labels):
#     conf_matrix[t, p] += 1

# Convert to NumPy for plotting
# cm = conf_matrix.numpy()

cm = np.array(
    [[14513, 301, 94, 51, 64, 602],
    [17, 13321, 0, 12, 12, 754],
    [56, 55, 29005, 128, 16, 158],
    [161, 38, 226, 61234, 233, 262],
    [112, 331, 1, 296, 74522, 133],
    [1726, 3792, 299, 553, 24, 13805]]
)

true_positives = np.diag(cm)
false_positives = np.sum(cm, axis=0) - true_positives
precision = true_positives / (true_positives + false_positives)

# Handle division by zero (if any class has no predictions)
precision = np.nan_to_num(precision)

print(precision)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7, 6))
cax = ax.matshow(cm, cmap=plt.cm.Blues)
plt.colorbar(cax)

# Add labels
classes = ['Tor', 'Non-Tor', 'Obfs4', 'Meek', 'FTE', 'WebTunnel'] # Modify if you have specific class names
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Annotate each cell with the count
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

# plt.title('Confusion Matrix')
plt.show()
