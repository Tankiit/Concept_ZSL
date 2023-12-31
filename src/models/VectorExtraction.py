import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, start_dim, intermediary_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(start_dim, intermediary_dim)
        self.fc2 = nn.Linear(intermediary_dim, start_dim)

        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.gelu(self.fc1(x))
        return self.fc2(out) + x

from vector_quantize_pytorch import VectorQuantize
class ResExtr(nn.Module):
    def __init__(self, start_dim, features, depth, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.features = features

        self.layers = nn.ModuleList([ResBlock(start_dim, start_dim) for _ in range(depth)])

        self.to_out = nn.Linear(start_dim, features)
        self.bin_quantize = VectorQuantize(
                            dim = 1,
                            codebook_size = 2,
                            freeze_codebook = True
                        )
        
        self.bin_quantize.codebook = torch.tensor([[[ 0.],
        [1.]]])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.to_out(x).view(-1, self.features, 1)

        quantize, _, commit_loss = self.bin_quantize(x)

        return quantize.view(-1, self.features), commit_loss

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FEATURES = 85
    CLASSES = 50

    model = ResExtr(2048, FEATURES, 1).to(device)
    print(model.bin_quantize._codebook.embed)