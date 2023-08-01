import pytorch_lightning as pl
import torch.nn as nn
import torch
from ResnetExtraction import ResExtr
from torchmetrics import Accuracy
from AwA2Loader import get_train_test_zsl
class ResnetExtractorLightning(pl.LightningModule):
    def __init__(self, num_features, depth, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_features = num_features
        self.num_classes = num_classes
        
        self.model = ResExtr(2048, num_features, depth)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=1)

    def forward(self, src):
        self.model.bin_quantize._codebook.embed = torch.tensor([[[ 0.],
         [1.]]], device="cuda")
        return self.model(src)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
    
    def loss_fn(self, out, y_predicates, y_classes, ft_weight=.1):
        out = out.view(-1, 1, self.num_features)
        ANDed = out * y_predicates
        diff = ANDed - out

        entr_loss = nn.CrossEntropyLoss()
        loss_cl = entr_loss(diff.sum(dim=2), y_classes)

        batch_size = out.shape[0]

        labels = torch.randint(0, self.num_classes-1, (batch_size, ), device="cuda")
        classes = torch.zeros(batch_size, self.num_classes, device="cuda")
        classes[torch.arange(batch_size), labels] = 1
        classes = classes.view(batch_size, self.num_classes, 1).expand(batch_size, self.num_classes, self.num_features)

        extra_features = out - y_predicates + (out - y_predicates).pow(2)

        loss_ft = torch.masked_select(extra_features, (1-classes).bool()).view(-1, self.num_features).sum() / batch_size

        return loss_cl + loss_ft * ft_weight * loss_cl/loss_ft
    
    def training_step(self, batch, batch_idx):
        features, labels = batch['features'], batch['labels']
        inputs = features.cuda()
        labels = labels.cuda()
    
        outputs, commit_loss = self.forward(inputs)

        loss = self.loss_fn(outputs, train_predicate_matrix, labels) + commit_loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, labels = batch['features'], batch['labels']
        inputs = features.cuda()
        labels = labels.cuda()
    
        outputs, commit_loss = self.forward(inputs)

        loss = self.loss_fn(outputs, test_predicate_matrix, labels) + commit_loss
        self.log('val_loss', loss, prog_bar=True)

        outputs = outputs.view(-1, 1, self.num_features)
        ANDed = outputs * test_predicate_matrix
        diff = ANDed - outputs
        acc = self.accuracy(diff.sum(dim=2), labels)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        return loss

    def val_dataloader(self):
        return torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=False, num_workers=4)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True, num_workers=4)

if __name__ == "__main__":
    feature_file = "datasets/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt"
    label_file = "datasets/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt"
    predicate_matrix_file = "datasets/Animals_with_Attributes2/predicate-matrix-binary.txt"
    train_classes_file = "datasets/Animals_with_Attributes2/Features/ResNet101/trainclasses.txt"
    classes_labels_file = "datasets/Animals_with_Attributes2/classes.txt"
    train_set, val_set, train_predicate_matrix, test_predicate_matrix = get_train_test_zsl(feature_file, label_file, predicate_matrix_file, classes_labels_file, train_classes_file)

    callbacks = []

    model = ResnetExtractorLightning(85, 1, 50)

    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=15, precision="16-mixed", callbacks=callbacks)

    #Create a Tuner
    from pytorch_lightning.tuner import Tuner
    tuner = Tuner(trainer)

    #finds learning rate automatically
    #sets hparams.lr or hparams.learning_rate to that learning rate
    tuner.lr_find(model)

    trainer.fit(model)
