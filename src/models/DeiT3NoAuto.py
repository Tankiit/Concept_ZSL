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
import timm
class ResExtr(nn.Module):
    def __init__(self, features, classes, deit_type=1, pretrained=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.features = features
        self.classes = classes

        if deit_type == 1:
            self.deit = timm.create_model("deit3_medium_patch16_224.fb_in22k_ft_in1k", pretrained=True)
            self.deit.head = nn.Linear(512, features)
        elif deit_type == 2:
            self.deit = timm.create_model("deit3_base_patch16_224.fb_in22k_ft_in1k", pretrained=True)
            self.deit.head = nn.Linear(768, features)
        else:
            self.deit = timm.create_model("deit3_large_patch16_384.fb_in22k_ft_in1k", pretrained=True)
            self.deit.head = nn.Linear(1024, features)
        
        self.bin_quantize = VectorQuantize(
                            dim = 1,
                            codebook_size = 2,
                            ema_update = False
                        )
        
        self.bin_quantize.codebook = torch.tensor([[ 0.],
        [1.]])
        
    def forward(self, x):
        x = self.deit(x).view(-1, self.features, 1)

        quantize, _, commit_loss = self.bin_quantize(x)

        return quantize.view(-1, self.features), commit_loss

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    FEATURES = 85
    CLASSES = 50

    model = ResExtr(FEATURES, CLASSES, deit_type=2, pretrained=True).to(device)
    
    inp = torch.randn(1, 3, 224, 224).to(device)
    print(model(inp)[0].shape)