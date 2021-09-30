import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import grad
import torch.optim as optim
import torch.autograd as autograd

from generator import Generator
from discriminator import Discriminator
from dataset import get_mnist_dataloaders

class Trainer():

    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iteration=5, print_every=50, use_cuda=True):
        
        self.G = generator
        self.G_optim = gen_optimizer
        self.D = discriminator
        self.D_optim = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'GN': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iteration
        self.print_every = print_every

        if self.use_cuda:

            self.G.cuda()
            self.D.cuda()

    def critic_train_iteration(self, data):

        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        data = data.cuda()

        d_real = self.D(data)
        d_generated = self.D(generated_data)

        gradient_penalty = self.calc_gp(data, generated_data)
        self.losses['GP'].append(gradient_penalty.item())

        self.D_optim.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()
        self.D_optim.step()

        self.losses['D'].append(d_loss.item())

    def generator_iter(self, data):

        self.G_optim.zero_grad()

        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size)

        d_generated = self.D(generated_data)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.G_optim.step()

        self.losses['G'].append(g_loss.item())

    def calc_gp(self, real_data, generated_data):

        batch_size = real_data.size()[0]
        alpha = torch.randn(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data).cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = interpolated.cuda()
        interpolated.requires_grad = True

        prob_interpolated = self.D(interpolated)

        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated, 
                                  grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                                  create_graph=True, retain_graph=True)[0]
        
        gradients = gradients.view(batch_size, -1)

        self.losses['GN'].append(gradients.norm(2, dim=1).mean().item())

        grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return self.gp_weight * ((grad_norm - 1) ** 2).mean()

    def train_epoch(self, data_loader):

        for i, data in enumerate(data_loader):

            self.num_steps += 1
            self.critic_train_iteration(data[0])

            if self.num_steps % self.critic_iterations == 0:
                #print(self.num_steps)
                self.generator_iter(data[0])

            if i % self.print_every == 0 and i != 0:

                print(f"[Step - {i + 1}] [Discriminator Loss - {self.losses['D'][-1]}] [Gradient Penalty - {self.losses['GP'][-1]}] [Generator Loss - {self.losses['G'][-1]}]")

    def train(self, data_loader, epochs, save_training_gif=True):

        if save_training_gif:

            fixed_latents = self.G.sample_latent(64)
            fixed_latents = fixed_latents.cuda()

            training_progress_images = []

        for epoch in range(epochs):

            print(f"\nEpoch {epoch + 1}")
            self.train_epoch(data_loader)

            if save_training_gif:

                img_grid = make_grid(self.G(fixed_latents).cpu().data)
                img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
                training_progress_images.append(img_grid)

            if save_training_gif:

                imageio.mimsave(f'/content/train_{epoch + 1}_epochs.gif', training_progress_images)

        

        imageio.mimsave(f'/content/train_{epoch + 1}_epochs.gif', training_progress_images)

    def sample_generator(self, num_samples):

        latent_samples = self.G.sample_latent(num_samples)

        if self.use_cuda:
            latent_samples = latent_samples.cuda()

        generated_data = self.G(latent_samples)

        return generated_data

