import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Example: Generating random true and predicted labels
# Replace these with your actual data
np.random.seed(42)  # For reproducibility
num_samples = 100
num_classes = 6

# Simulating true labels and predictions
y_true = np.random.randint(0, num_classes, num_samples)
y_pred = np.random.randint(0, num_classes, num_samples)

# Generate the confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

# Visualize the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Class {i}" for i in range(num_classes)])
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Printing the confusion matrix
print("Confusion Matrix:\n", cm)
