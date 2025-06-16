#!/usr/bin/python
# -*- coding: utf-8 -*-
from torch import nn
from torchvision.datasets import ImageFolder
from de_resnet import de_wide_resnet50_2
import torch.nn.functional as F
import torch
class autoencoder(nn.Module):
    def __init__(self, out_channels):
        super(autoencoder, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
        # encoder
        nn.Conv2d(in_channels= 3, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1)
        )
        self.encoderD=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8)


        # self.vq_layer = VectorQuantizer(64, 64, 1)

        # 解码器部分
        # decoder
        self.decoderU = nn.Sequential(
        nn.Upsample(size=3, mode='bicubic'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bicubic'),
        )
        self.decoder = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bicubic'),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bicubic'),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=64, mode='bicubic'),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )

    def forward(self, x):
        # 编码器
        x1 = self.encoder(x)

        # VQx, loss = self.vq_layer(x1)

        x2 = self.encoderD(x1)

        # 解码器
        x3 = self.decoderU(x2)

        # x4 = torch.cat([x3,  VQx], 1)

        x4 = self.decoder(x3)


        return x4
class VectorQuantizer(nn.Module):


    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

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
            # quantized = quantized.flatten(2).transpose(1, 2).contiguous()  # B H*W C

            loss=0

            return quantized,loss

        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        # quantized = quantized.permute(0, 3, 1, 2).contiguous()
        # quantized = quantized.flatten(2).transpose(1, 2).contiguous()  # B H*W C
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
def get_autoencoder(out_channels=384):

    return nn.Sequential(
        # encoder
        # nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
        #           padding=1),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
        #           padding=1),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(in_channels=out_channels, out_channels=64, kernel_size=4, stride=2,
        #           padding=1),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
        #           padding=1),
        # nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=64, kernel_size=16),
        # decoder
        nn.Upsample(size=3, mode='bicubic'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bicubic'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bicubic'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bicubic'),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bicubic'),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=64, mode='bicubic'),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )

def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )
def get_pdn_Smedium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )
def get_pdn_medium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )

class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
def AE(out_channels):
    return autoencoder(out_channels)

def StudentN(out_channels):
    return Student(out_channels)

class Student(nn.Module):
    def __init__(self, out_channels):
        super(Student, self).__init__()
        pad_mult = 1
        # 编码器部分
        self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )

        self.decoder=nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=512, kernel_size=4, stride=2,
                  padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2,
                      padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoderM = nn.Sequential(
            nn.Upsample(size=32, mode='bicubic'),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1,
                      padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=64, mode='bicubic'),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1)
        )

        self.decoderAEG = nn.Sequential(
            # encoder
            # nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
            #           padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
            #           padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=out_channels, out_channels=64, kernel_size=4, stride=2,
            #           padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
            #           padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=16),
            # decoder
            nn.Upsample(size=4, mode='bicubic'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=8, mode='bicubic'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=32, mode='bicubic'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=64, mode='bicubic'),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      stride=1, padding=1)
        )

    def forward(self, x):
        # 编码器

        x1 = self.encoder(x)


        # 解码器
        x2 = self.decoder(x1)



        x3=self.decoderM(x2)


        x4 = self.decoderAEG(x2)

        return x1,x4,x3,