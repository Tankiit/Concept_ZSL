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
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights
class ResExtr(nn.Module):
    def __init__(self, features, classes, pretrained=False, resnet_type=18, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.features = features
        self.classes = classes

        if resnet_type == 18:
            if pretrained:
                self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
                self.resnet.fc = nn.Linear(512, features)
            else:
                self.resnet = resnet18()
                self.resnet.fc = nn.Linear(512, features)
        elif resnet_type == 34:
            if pretrained:
                self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
                self.resnet.fc = nn.Linear(1024, features)
            else:
                self.resnet = resnet34()
                self.resnet.fc = nn.Linear(1024, features)
        elif resnet_type == 50:
            if pretrained:
                self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
                self.resnet.fc = nn.Linear(2048, features)
            else:
                self.resnet = resnet50()
                self.resnet.fc = nn.Linear(2048, features)
        elif resnet_type == 101:
            if pretrained:
                self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
                self.resnet.fc = nn.Linear(2048, features)
            else:
                self.resnet = resnet101()
                self.resnet.fc = nn.Linear(2048, features)
        elif resnet_type == 152:
            if pretrained:
                self.resnet = resnet152(weights=ResNet152_Weights.DEFAULT)
                self.resnet.fc = nn.Linear(2048, features)
            else:
                self.resnet = resnet152()
                self.resnet.fc = nn.Linear(2048, features)
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

    model = ResExtr(2048, FEATURES, pretrained=True, resnet_type=152).to(device)
    print(model.bin_quantize._codebook.embed)