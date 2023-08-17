import pytorch_lightning as pl
import torch.nn as nn
import torch
import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/models")
from VectorExtraction import ResExtr
from torchmetrics import Accuracy
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
        return self.model(src)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
    
    def loss_fn(self, out, predicate_matrix, labels, ft_weight=1., pos_ft_weight=0.):
        out = out.view(-1, 1, self.num_features) # out is a batch of 1D binary vectors
        ANDed = out * predicate_matrix # AND operation
        diff = ANDed - out # Difference of ANDed and out => if equal, then out is a subset of its class' predicates

        entr_loss = nn.CrossEntropyLoss()
        loss_cl = entr_loss(diff.sum(dim=2), labels) # Is "out" a subset of its class' predicates?

        batch_size = out.shape[0]

        classes = torch.zeros(batch_size, self.num_classes, device="cuda")
        classes[torch.arange(batch_size), labels] = 1
        classes = classes.view(batch_size, self.num_classes, 1).expand(batch_size, self.num_classes, self.num_features)

        extra_features = out - predicate_matrix + (out - predicate_matrix).pow(2)

        loss_neg_ft = torch.masked_select(extra_features, (1-classes).bool()).view(-1, self.num_features).sum() / batch_size

        labels_predicate = predicate_matrix[labels]
        extra_features_in = torch.masked_select(extra_features, classes.bool()).view(-1, self.num_features)
        loss_pos_ft = (labels_predicate - out.view(batch_size, self.num_features) + extra_features_in/2).sum() / batch_size

        return loss_cl + loss_neg_ft * ft_weight * loss_cl.item()/loss_neg_ft.item() + loss_pos_ft * pos_ft_weight * loss_cl.item()/loss_pos_ft.item()

    def training_step(self, batch, batch_idx):
        features, labels = batch['features'], batch['labels']
        inputs = features.cuda()
        labels = labels.cuda()
    
        outputs, commit_loss = self.forward(inputs)

        loss = self.loss_fn(outputs, predicate_matrix, labels) + commit_loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features, labels = batch['features'], batch['labels']
        inputs = features.cuda()
        labels = labels.cuda()
    
        outputs, commit_loss = self.forward(inputs)

        loss = self.loss_fn(outputs, predicate_matrix, labels) + commit_loss
        self.log('val_loss', loss, prog_bar=True)

        outputs = outputs.view(-1, 1, self.num_features)
        ANDed = outputs * predicate_matrix
        diff = ANDed - outputs
        acc = self.accuracy(diff.sum(dim=2), labels)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        return loss

    def val_dataloader(self):
        return torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)

if __name__ == "__main__":
    from numpy import loadtxt
    predicate_matrix_file = "datasets/Animals_with_Attributes2/predicate-matrix-binary.txt"
    predicate_matrix = torch.from_numpy(loadtxt(predicate_matrix_file, dtype=int)).cuda()
    import pickle
    # pickle val_set
    with open("val_set.pkl", "rb") as f:
        val_set = pickle.load(f)

    # pickle train_set
    with open("train_set.pkl", "rb") as f:
        train_set = pickle.load(f)

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
