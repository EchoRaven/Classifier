import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import json
import cv2


class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        # [B, H, W, C] -> [BHW, C]
        flat_x = x.reshape(-1, self.embedding_dim)

        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)  # [B, H, W, C]

        if not self.training:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            return quantized

        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(self.embeddings.weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, self.embeddings.weight.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)


class Encoder(nn.Module):
    """Encoder of VQ-VAE"""

    def __init__(self, in_dim=3, latent_dim=16):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, 1),
        )

    def forward(self, x):
        return self.convs(x)


class Decoder(nn.Module):
    """Decoder of VQ-VAE"""

    def __init__(self, out_dim=1, latent_dim=16):
        super().__init__()
        self.out_dim = out_dim
        self.latent_dim = latent_dim

        self.convs = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_dim, 3, padding=1),
        )

    def forward(self, x):
        return self.convs(x)


class VQVAE(nn.Module):
    """VQ-VAE"""

    def __init__(self, in_dim, embedding_dim, num_embeddings, data_variance,
                 commitment_cost=0.25):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.data_variance = data_variance

        self.encoder = Encoder(in_dim, embedding_dim)
        self.vq_layer = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)
        self.decoder = Decoder(in_dim, embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        if not self.training:
            e = self.vq_layer(z)
            x_recon = self.decoder(e)
            return e, x_recon

        e, e_q_loss = self.vq_layer(z)
        x_recon = self.decoder(e)

        recon_loss = F.mse_loss(x_recon, x) / self.data_variance

        return e_q_loss + recon_loss


class VQVAEModel(nn.Module):
    def __init__(self, in_dim, embedding_dim, num_embeddings, data_variance, commitment_cost=0.25):
        super(VQVAEModel, self).__init__()
        self.model = VQVAE(in_dim, embedding_dim, num_embeddings, data_variance, commitment_cost)
        self.model.cuda()

    def Train(self,
              train_loader,
              epochs,
              print_freq,
              encoder_name,
              decoder_name,
              vq_name,
              log_name,
              lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        log = []
        for epoch in range(epochs):
            print("Start training epoch {}".format(epoch, ))
            for i, (images, labels) in enumerate(train_loader):
                images = images - 0.5  # normalize to [-0.5, 0.5]
                images = images.cuda()
                loss = self.model(images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
                    print("\t [{}/{}]: loss {}".format(i, len(train_loader), loss.item()))
                    log.append({"loss": loss.item()})

        f = open(log_name, 'w', encoding='utf-8')
        f.write(json.dumps(log, ensure_ascii=False, indent=4))
        f.close()
        torch.save(self.encoder, encoder_name)
        torch.save(self.decoder, decoder_name)
        torch.save(self.vq_layer, vq_name)

    def forward(self, img):
        return self.model(img)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset1 = datasets.MNIST('data/', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('data/', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=128)

    # compute the variance of the whole training set to normalise the Mean Squared Error below.
    train_images = []
    for images, labels in train_loader:
        train_images.append(images)
    train_images = torch.cat(train_images, dim=0)
    train_data_variance = torch.var(train_images)

    model = VQVAEModel(1, 16, 128, train_data_variance)
    model = model.cuda()
    model.Train(train_loader, 300, 500, 'encoder.pth', 'decoder.pth', 'vq.pth', 'log.json', 1e-3)

