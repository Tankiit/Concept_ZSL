import torch
from vector_quantize_pytorch import VectorQuantize

class BSSLoss(torch.nn.Module):
    def __init__(self, n_features, use_loss_ft=True, add_predicate_matrix=False, n_classes=None, ft_weight=1, mean_attr_weight=2, eps=1e-10, pre_quantized=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_features = n_features
        self.ft_weight = ft_weight
        self.mean_attr_weight = mean_attr_weight
        self.pre_quantized = pre_quantized
        self.use_loss_ft = use_loss_ft

        self.eps = eps

        self.ent_loss = torch.nn.CrossEntropyLoss()

        if add_predicate_matrix:
            assert n_classes is not None, "If you want to add the predicate matrix in the loss, you need to specify the number of classes"
            self.predicate_matrix = torch.nn.Parameter(torch.randn(n_classes, n_features))
            self.n_classes = n_classes

        if not self.pre_quantized or add_predicate_matrix:
            self.bin_quantize = VectorQuantize(
                            dim = 1,
                            codebook_size = 2,
                            ema_update = False
                        )

            self.bin_quantize.codebook = torch.tensor([[ 0.], [1.]])

    def add_predicate_matrix_to_optimizer(self, optimizer):
        assert hasattr(self, 'predicate_matrix'), "You need to have the predicate matrix in the loss before adding it to the optimizer"
        optimizer.param_groups.append({'params': self.predicate_matrix})

    def get_predicate_matrix(self):
        assert hasattr(self, 'predicate_matrix'), "You need to have the predicate matrix in the loss before getting it"
        
        predicate_matrix, _, _ = self.bin_quantize(self.predicate_matrix.view(self.n_classes * self.n_features, 1))
        return predicate_matrix.view(self.n_classes, self.n_features)
    
    def soft_threshold(self, x):
        return torch.clamp(x, min=0) * torch.clamp(torch.abs(x)+1, max=1)
    
    def binarize_output(self, x):
        if not self.pre_quantized:
            x, _, _ = self.bin_quantize(x.view(-1, self.n_features, 1))
        x = x.view(-1, self.n_features)
        return x

    def __call__(self, x, labels, predicate_matrix=None):
        if not self.pre_quantized:
            x, _, commit_loss = self.bin_quantize(x.view(-1, self.n_features, 1))

        if hasattr(self, 'predicate_matrix'):
            predicate_matrix, _, commit_loss2 = self.bin_quantize(self.predicate_matrix.view(self.n_classes * self.n_features, 1))
            predicate_matrix = predicate_matrix.view(self.n_classes, self.n_features)
        else:
            assert predicate_matrix is not None, "If you don't have the predicate matrix in the loss, you need to specify it"

            if not self.pre_quantized:
                predicate_matrix, _, commit_loss2 = self.bin_quantize(predicate_matrix.view(self.n_classes * self.n_features, 1))
                predicate_matrix = predicate_matrix.view(self.n_classes, self.n_features)

        x = x.view(-1, 1, self.n_features)

        ANDed = x * predicate_matrix
        
        diff = ANDed - x
        
        loss_cl = self.ent_loss(diff.sum(dim=2), labels)

        batch_size = x.shape[0]

        x = x.view(-1, self.n_features)
        diff_square = (x - predicate_matrix[labels]).pow(2)

        false_positives = (x - predicate_matrix[labels] + diff_square).sum() / batch_size
        missing_attr = (predicate_matrix[labels] - x + diff_square).sum() / batch_size

        loss_mean_attr = (predicate_matrix.sum(dim=1).mean() - self.n_features//2).pow(2)

        loss_ft = self.mean_attr_weight*loss_mean_attr + false_positives + missing_attr
        loss_ft *= loss_cl.item()/(loss_ft.item() + self.eps)
        loss_ft = loss_ft if self.use_loss_ft else 0

        if hasattr(self, 'predicate_matrix') and not self.pre_quantized:
            return loss_cl + loss_ft * self.ft_weight + commit_loss + commit_loss2
        elif hasattr(self, 'predicate_matrix'):
            return loss_cl + loss_ft * self.ft_weight + commit_loss2
        elif not self.pre_quantized:
            return loss_cl + loss_ft * self.ft_weight + commit_loss + commit_loss2
        else:
            return loss_cl + loss_ft * self.ft_weight
        
import torch.nn as nn
class Packed(nn.Module):
    def __init__(self, model, BSSLossObj) -> None:
        super().__init__()
        self.model = model
        self.predicate_matrix = nn.Parameter(BSSLossObj.get_predicate_matrix())
        self.n_features = BSSLossObj.n_features

        self.bin_quantize = VectorQuantize(
                            dim = 1,
                            codebook_size = 2,
                            ema_update = False
                        )
        self.bin_quantize.codebook = torch.tensor([[ 0.], [1.]])

    def get_features(self, x):
        x, _, _ = self.bin_quantize(self.model(x).view(-1, self.n_features, 1))
        return x.view(-1, self.n_features)

    def forward(self, x):
        x = self.get_features(x).view(-1, 1, self.n_features)

        ANDed = x * self.predicate_matrix
        
        diff = ANDed - x
        
        return diff.sum(dim=2)
    
def pack_model(model, BSSLossObj):
    return Packed(model, BSSLossObj)

if __name__ == "__main__":
    loss_fn = BSSLoss(64)
    
    a = torch.randn(5)
    print(a)
    print(loss_fn.soft_threshold(a))