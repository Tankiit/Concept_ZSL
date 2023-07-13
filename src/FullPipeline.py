from VQVAE import Encoder, Decoder
from FeatureExtractor import Extractor
import torch.nn as nn

class FullPipeline(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, s_size, codebook_size, attn_dim, attn_depth, attn_dim_head, max_features, codebook_dim):
        super().__init__()

        self.encoder = Encoder(in_channel, channel, n_res_block, n_res_channel, s_size, codebook_size)
        self.decoder = Decoder(channel, in_channel, channel, n_res_block, n_res_channel, s_size)

        final_channel = channel // (2**s_size)
        self.extractor = Extractor(final_channel**2, max_features, codebook_size, channel, codebook_dim, attn_dim, attn_depth, attn_dim_head)

    def forward(self, input):
        quantized, commit_loss1 = self.encoder(input)
        print(quantized.shape)
        recon = self.decoder(quantized)
        features, commit_loss2 = self.extractor(quantized)

        return recon, features, commit_loss1, commit_loss2
    
if __name__ == "__main__":
    import torch
    img = torch.randn(1, 3, 256, 256)

    IN_CHANNEL = 3
    CHANNEL = 128
    N_RES_BLOCK = 2
    N_RES_CHANNEL = 32
    S_SIZE = 5
    CODEBOOK_SIZE = 512
    ATTN_DIM = 128
    ATTN_DEPTH = 6
    ATTN_DIM_HEAD = 16
    MAX_FEATURES = 39
    CODEBOOK_DIM = 256

    model = FullPipeline(IN_CHANNEL, CHANNEL, N_RES_BLOCK, N_RES_CHANNEL, S_SIZE, CODEBOOK_SIZE, ATTN_DIM, ATTN_DEPTH, ATTN_DIM_HEAD, MAX_FEATURES, CODEBOOK_DIM)
    recon, features, commit_loss1, commit_loss2 = model(img)
    print(recon.shape, features.shape, commit_loss1, commit_loss2)