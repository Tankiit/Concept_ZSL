import torch
import numpy as np
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

        self.layers = nn.ModuleList([ResBlock(start_dim, start_dim*2) for _ in range(depth)])

        self.to_out = nn.Linear(start_dim, features)
        self.bin_quantize = VectorQuantize(
                            dim = 1,
                            codebook_size = 2,
                        )
        
        self.bin_quantize._codebook.embed = torch.tensor([[[ 0.],
         [1.]]], requires_grad=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.to_out(x).view(-1, self.features, 1)
        quantize, _, commit_loss = self.bin_quantize(x)

        return quantize.view(-1, self.features), commit_loss
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_features = np.loadtxt('datasets/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt', dtype=np.float32)
resnet_features = torch.from_numpy(resnet_features).to(device)
print(resnet_features.shape)

FEATURES = 85
CLASSES = 50

model = ResExtr(2048, FEATURES, 4).to(device)
out = model(resnet_features)[0]
print(out.shape)

class_features = torch.randint(0, 2, (50, FEATURES))
print(class_features.shape)

ANDed = out.view(-1, 1, FEATURES) * class_features
print(ANDed.shape)

def loss_fn(out, y_features, y_classes, ft_weight=1.0):
    out = out.view(-1, 1, FEATURES)
    ANDed = out * y_features
    diff = ANDed - out

    entr_loss = nn.CrossEntropyLoss()
    loss_cl = entr_loss(diff.sum(dim=2), y_classes)

    batch_size = out.shape[0]

    labels = torch.randint(0, CLASSES-1, (batch_size, ))
    classes = torch.zeros(batch_size, CLASSES)
    classes[torch.arange(batch_size), labels] = 1
    classes = classes.view(batch_size, CLASSES, 1).expand(batch_size, CLASSES, FEATURES)

    extra_features = out - y_features + (out - y_features).pow(2)
    loss_ft = torch.masked_select(extra_features, (1-classes).bool()).view(-1, FEATURES).sum()

    return loss_cl + loss_ft * ft_weight
