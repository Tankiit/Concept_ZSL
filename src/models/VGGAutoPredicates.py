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
from resnet import ResNet18, ResNet34, ResNet50
from torchvision.models import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
class ResExtr(nn.Module):
    def __init__(self, features, classes, vgg_type=16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.features = features
        self.classes = classes

        if vgg_type == 11:
            self.vgg = vgg11_bn(weights='DEFAULT')
            self.vgg.classifier[-1] = nn.Linear(4096, features)
        elif vgg_type == 13:
            self.vgg = vgg13_bn(weights='DEFAULT')
            self.vgg.classifier[-1] = nn.Linear(4096, features)
        elif vgg_type == 16:
            self.vgg = vgg16_bn(weights='DEFAULT')
            self.vgg.classifier[-1] = nn.Linear(4096, features)
        elif vgg_type == 19:
            self.vgg = vgg19_bn(weights='DEFAULT')
            self.vgg.classifier[-1] = nn.Linear(4096, features)
        else:
            raise ValueError("Unavailable resnet type")
        
        self.bin_quantize = VectorQuantize(
                            dim = 1,
                            codebook_size = 2,
                            ema_update = False
                        )
        
        self.bin_quantize.codebook = torch.tensor([[ 0.],
        [1.]])
        
        self.predicate_matrix = nn.Parameter(torch.randn(classes, features))
        
    def forward(self, x):
        x = self.vgg(x).view(-1, self.features, 1)

        quantize, _, commit_loss = self.bin_quantize(x)

        predicate_matrix, _, commit_loss2 = self.bin_quantize(self.predicate_matrix.view(self.classes * self.features, 1))
        predicate_matrix = predicate_matrix.view(self.classes, self.features)

        return quantize.view(-1, self.features), commit_loss + commit_loss2, predicate_matrix

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FEATURES = 85
    CLASSES = 50

    model = ResExtr(FEATURES, CLASSES, 19).to(device)
    
    input = torch.randn((64, 3, 256, 256)).to(device)
    print(model(input)[0].shape)