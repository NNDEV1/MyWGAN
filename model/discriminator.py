import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, img_size, dim):
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.image_to_features = nn.Sequential(
            nn.Conv2d(self.img_size[2], dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * 2, dim * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * 4, dim * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
        )

        output_size = 8 * dim * int(img_size[0] / 16) * int(img_size[1] / 16)

        self.features_to_prob = nn.Sequential(
            nn.Linear(output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        batch_size = x.size()[0]
        x = self.image_to_features(x)
        x = x.view(batch_size, -1)

        return self.features_to_prob(x)
