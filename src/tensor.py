import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Tanh()  # Output is in the range (-1, 1) for image data normalization
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_size),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, x):
        return self.model(x)


# Hyperparameters
input_size = 784  # 28x28 images
hidden_dim = 256
latent_size = 100  # Random noise vector size
batch_size = 64
learning_rate = 0.0002
num_epochs = 100

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to (-1, 1)
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize generator and discriminator
G = Generator(latent_size, hidden_dim, input_size)
D = Discriminator(input_size, hidden_dim, 1)

# Loss and optimizer
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer_G = optim.Adam(G.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(D.parameters(), lr=learning_rate)


# Function to create noise for generator
def create_noise(batch_size, latent_size):
    return torch.randn(batch_size, latent_size)


# Training Loop
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):

        # Flatten MNIST images into a vector
        images = images.view(-1, 28 * 28)
        size = images.shape[0]

        # Create real and fake labels
        real_labels = torch.ones(size, 1)
        fake_labels = torch.zeros(size, 1)

        # =======================
        # Train Discriminator
        # =======================
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        noise = create_noise(size, latent_size)
        fake_images = G(noise)
        outputs = D(fake_images.detach())  # detach to avoid backprop through generator
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # =======================
        # Train Generator
        # =======================
        noise = create_noise(size, latent_size)
        fake_images = G(noise)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)  # Try to trick D into thinking generated images are real

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if (i + 1) % 200 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, D(x): {real_score.mean().item()}, D(G(z)): {fake_score.mean().item()}')

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Generate and visualize some images
noise = create_noise(batch_size, latent_size)
fake_images = G(noise)
fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
fake_images = denorm(fake_images)

grid = torchvision.utils.make_grid(fake_images, nrow=8, padding=2, normalize=False)
plt.imshow(np.transpose(grid.detach().numpy(), (1, 2, 0)))
plt.show()
