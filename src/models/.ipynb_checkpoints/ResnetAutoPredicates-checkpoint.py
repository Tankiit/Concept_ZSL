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
from torchvision.models import resnet18, ResNet18_Weights
class ResExtr(nn.Module):
    def __init__(self, features, classes, pretrained=False, resnet_type=18, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.features = features
        self.classes = classes

        if pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.resnet.fc = nn.Linear(512, features)
        elif resnet_type == 18:
            self.resnet = ResNet18(num_classes=features)
        elif resnet_type == 34:
            self.resnet = ResNet34(num_classes=features)
        elif resnet_type == 50:
            self.resnet = ResNet50(num_classes=features)
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
        x = self.resnet(x).view(-1, self.features, 1)

        quantize, _, commit_loss = self.bin_quantize(x)

        predicate_matrix, _, commit_loss2 = self.bin_quantize(self.predicate_matrix.view(self.classes * self.features, 1))
        predicate_matrix = predicate_matrix.view(self.classes, self.features)

        return quantize.view(-1, self.features), commit_loss + commit_loss2, predicate_matrix

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FEATURES = 85
    CLASSES = 50

    model = ResExtr(2048, FEATURES).to(device)
    print(model.bin_quantize._codebook.embed)