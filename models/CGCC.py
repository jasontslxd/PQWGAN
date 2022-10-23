import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalGAN():
    def __init__(self, image_size, channels, latent_size):
        self.image_shape = (channels, image_size, image_size)
        self.critic = self.ClassicalCritic(self.image_shape)
        self.generator = self.ClassicalGenerator(latent_size, self.image_shape)

    class ClassicalGenerator(nn.Module):
        def __init__(self, latent_size, image_shape):
            super().__init__()
            self.latent_size = latent_size
            self.image_shape = image_shape

            self.fc1 = nn.Linear(latent_size, 256)
            self.fc2 = nn.Linear(256, 512)
            self.fc3 = nn.Linear(512, 1024)
            self.fc4 = nn.Linear(1024, int(np.prod(self.image_shape)))

        def forward(self, x):
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            x = F.leaky_relu(self.fc3(x), 0.2)
            x = torch.tanh(self.fc4(x))
            x = x.view(x.shape[0], *self.image_shape)
            return x

    class ClassicalCritic(nn.Module):
        def __init__(self, image_shape):
            super().__init__()
            self.image_shape = image_shape

            self.fc1 = nn.Linear(int(np.prod(self.image_shape)), 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 1)

        def forward(self, x):
            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            return self.fc3(x)