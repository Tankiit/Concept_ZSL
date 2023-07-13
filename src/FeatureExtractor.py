import torch
import torch.nn as nn
from x_transformers import ContinuousTransformerWrapper, Encoder
from vector_quantize_pytorch import VectorQuantize

class Extractor(nn.Module):
    def __init__(self, input_len, max_features, codebook_size, in_dim, codebook_dim, attn_dim, attn_depth, attn_dim_head) -> None:
        super().__init__()

        heads = attn_dim // attn_dim_head

        self.transformer = ContinuousTransformerWrapper(
            dim_in = in_dim,
            dim_out = codebook_dim,
            max_seq_len = input_len + max_features,
            attn_layers = Encoder(
                dim = attn_dim,
                depth = attn_depth,
                heads = heads,
                attn_flash = True, # just set this to True if you have pytorch 2.0 installed
                #alibi_pos_bias = True, # turns on ALiBi positional embedding => doesn't work right now
            )
        )

        self.quantizer = VectorQuantize(
                            dim = codebook_dim,
                            codebook_size = codebook_size,
                            decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
                            commitment_weight = 1.   # the weight on the commitment loss
                        )
        
    def forward(self, input):
        x = self.transformer(input)
        quantized, _, commit_loss = self.quantizer(x)
        return quantized.sum(dim=1), commit_loss
    
if __name__ == "__main__":
    INPUT_LEN = 16 # latent img shape: torch.Size([1, 16, 128])
    MAX_FEATURES = 39 # max number of features (39 for AwA2)
    CODEBOOK_SIZE = 512 # number of codebook vectors
    IN_DIM = 128 # dimension of the input
    CODEBOOK_DIM = 256 # dimension of each codebook vector
    ATTN_DIM = 128 # dimension of the transformer
    ATTN_DEPTH = 6 # depth of the transformer
    ATTN_DIM_HEAD = 16 # dimension of each head of the transformer

    extractor = Extractor(INPUT_LEN, MAX_FEATURES, CODEBOOK_SIZE, IN_DIM, CODEBOOK_DIM, ATTN_DIM, ATTN_DEPTH, ATTN_DIM_HEAD)
    input = torch.randn(1, 16, 128)
    quantized, commit_loss = extractor(input)
    print(quantized.shape)