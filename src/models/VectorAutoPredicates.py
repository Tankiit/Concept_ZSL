import torch
import torch.nn as nn

from vector_quantize_pytorch import VectorQuantize
class ResExtr(nn.Module):
    def __init__(self, start_dim, features, classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.features = features
        self.classes = classes

        self.layers = nn.Sequential(
                nn.Linear(start_dim, 2048),
                nn.GELU(),
                nn.Linear(2048, features)
            )

        self.bin_quantize = VectorQuantize(
                dim = 1,
                codebook_size = 2,
                ema_update = False
            )
        
        self.bin_quantize.codebook = torch.tensor([[ 0.],
        [1.]])
        
        self.predicate_matrix = nn.Parameter(torch.randn(classes, features))
        
    def forward(self, x):
        x = self.layers(x).view(-1, self.features, 1)

        quantize, _, commit_loss = self.bin_quantize(x)

        predicate_matrix, _, commit_loss2 = self.bin_quantize(self.predicate_matrix.view(self.classes * self.features, 1))
        predicate_matrix = predicate_matrix.view(self.classes, self.features)

        return quantize.view(-1, self.features), commit_loss + commit_loss2, predicate_matrix

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FEATURES = 85
    CLASSES = 50

    model = ResExtr(2048, FEATURES, 1).to(device)
    print(model.bin_quantize._codebook.embed)