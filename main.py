from generator import Generator
from discriminator import Discriminator
from dataloader import get_mnist_dataloaders

import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import grad
import torch.optim as optim
import torch.autograd as autograd

data_loader, _ = get_mnist_dataloaders(batch_size=64)
img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=100, dim=16)
discriminator = Discriminator(img_size=img_size, dim=16)

lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
epochs = 200
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer)

trainer.train(data_loader, epochs, save_training_gif=True)
