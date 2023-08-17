import torch
import torch.nn as nn
from torchvision.models import resnet50

from vector_quantize_pytorch import VectorQuantize
class ResExtr(nn.Module):
    def __init__(self, start_dim, features, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.features = features

        self.resnet = resnet50(weights=None)
        self.resnet.fc = nn.Identity()

        self.to_out = nn.Linear(start_dim, features)
        self.bin_quantize = VectorQuantize(
                            dim = 1,
                            codebook_size = 2,
                            freeze_codebook = True
                        )
        
        self.bin_quantize.codebook = torch.tensor([[ 0.],
        [1.]])
        
    def forward(self, x):
        x = self.resnet(x)

        x = self.to_out(x).view(-1, self.features, 1)

        quantize, _, commit_loss = self.bin_quantize(x)

        return quantize.view(-1, self.features), commit_loss

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FEATURES = 85
    CLASSES = 50

    model = ResExtr(2048, FEATURES).to(device)

    input = torch.rand((1, 3, 256, 256)).to(device)
    print(model(input))