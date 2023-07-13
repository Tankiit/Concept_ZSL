import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

# https://github.com/aisinai/vqvae2/blob/master/networks.py
class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, s_size, codebook_size):
        super().__init__()

        blocks = []

        for i in range(s_size):
            divider = 2**(s_size-i)
            blocks.extend([
                nn.Conv2d(in_channel, channel // divider, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            in_channel = channel // divider

        blocks.extend([
            nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
        ])

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

        self.quantizer = VectorQuantize(
                            dim = channel,
                            codebook_size = codebook_size,
                            decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
                            commitment_weight = 1.   # the weight on the commitment loss
                        )

    def forward(self, input):
        x = self.blocks(input)
        x = x.view(x.shape[0], -1, x.shape[1])
        quantized, _, commit_loss = self.quantizer(x)
        return quantized, commit_loss

import math
class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, s_size):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        for i in range(1, s_size+1):
            divider = 2**i
            blocks.extend([
                nn.ConvTranspose2d(in_channel, channel // divider, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            in_channel = channel // divider

        blocks.extend([
            nn.ConvTranspose2d(in_channel, out_channel, 4, stride=2, padding=1)
        ])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        dim = math.isqrt(input.shape[1])
        x = input.view(input.shape[0], input.shape[2], dim, dim)
        return self.blocks(x)

if __name__ == '__main__':
    enc = Encoder(3, 128, 2, 32, 5, 512)

    img = torch.randn(1, 3, 256, 256)
    out = enc(img)
    print(out[0].shape)

    dec = Decoder(128, 3, 128, 2, 32, 5)
    out = dec(out[0])
    print(out.shape)