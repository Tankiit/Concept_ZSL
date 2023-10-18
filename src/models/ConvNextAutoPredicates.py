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
from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large
class ResExtr(nn.Module):
    def __init__(self, features, classes, convnext_type=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.features = features
        self.classes = classes

        if convnext_type == 1:
            self.convnext = convnext_tiny(weights='DEFAULT')
            self.convnext.classifier[-1] = nn.Linear(768, features)
        elif convnext_type == 2:
            self.convnext = convnext_small(weights='DEFAULT')
            self.convnext.classifier[-1] = nn.Linear(768, features)
        elif convnext_type == 3:
            self.convnext = convnext_base(weights='DEFAULT')
            self.convnext.classifier[-1] = nn.Linear(1024, features)
        elif convnext_type == 4:
            self.convnext = convnext_large(weights='DEFAULT')
            self.convnext.classifier[-1] = nn.Linear(1536, features)
        else:
            raise ValueError("Unavailable convnext type")
        
        self.bin_quantize = VectorQuantize(
                            dim = 1,
                            codebook_size = 2,
                            ema_update = False
                        )
        
        self.bin_quantize.codebook = torch.tensor([[ 0.],
        [1.]])
        
        self.predicate_matrix = nn.Parameter(torch.randn(classes, features))
        
    def forward(self, x):
        x = self.convnext(x).view(-1, self.features, 1)

        quantize, _, commit_loss = self.bin_quantize(x)

        predicate_matrix, _, commit_loss2 = self.bin_quantize(self.predicate_matrix.view(self.classes * self.features, 1))
        predicate_matrix = predicate_matrix.view(self.classes, self.features)

        return quantize.view(-1, self.features), commit_loss + commit_loss2, predicate_matrix

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FEATURES = 85
    CLASSES = 50

    model = ResExtr(FEATURES, CLASSES).to(device)
    
    input = torch.randn((64, 3, 256, 256)).to(device)
    print(model(input)[0].shape)