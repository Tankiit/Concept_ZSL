import lightning as L
import timm, torch, sys
sys.path.insert(0, "/".join(__file__.split("/")[:-2]))
from SubsetLoss import BSSLoss
from torchmetrics import Accuracy
from sam import SAM
class BSSTrainer(L.LightningModule):
    def __init__(self, model, NUM_CLASSES, NUM_FEATURES, EPOCHS, training_loader, validation_loader, NUM_EXCLUDE=0):
        super().__init__()
        self.save_hyperparameters()

        self.NUM_FEATURES = NUM_FEATURES
        self.training_loader = training_loader
        self.validation_loader = validation_loader

        self.model = model

        self.criterion = BSSLoss(NUM_FEATURES, add_predicate_matrix=True, n_classes=NUM_CLASSES - NUM_EXCLUDE)
        self.zerocrit = torch.nn.MSELoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES - NUM_EXCLUDE, top_k=1)

        optimizer = self.configure_optimizers()
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-4, epochs=EPOCHS, steps_per_epoch=len(training_loader))

        self.running_false_positives = 0
        self.running_missing_attr = 0
        self.running_out_attributes = 0
        self.step_count = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.running_false_positives = 0
        self.running_missing_attr = 0
        self.running_out_attributes = 0
        self.step_count = 0

        inputs, labels, zero_out = batch["train_images"], batch["train_labels"], batch["zero_images"]
        outputs = self.model(inputs)
        zero_outputs = self.criterion.binarize_output(self.model(zero_out))
        loss = self.criterion(outputs, labels)
        loss += self.zerocrit(zero_outputs, torch.zeros_like(zero_outputs))

        opt = self.optimizers()
        
        loss.backward()
        opt.first_step(zero_grad=True)

        outputs = self.model(inputs)
        zero_outputs = self.criterion.binarize_output(self.model(zero_out))
        loss = self.criterion(outputs, labels)
        loss += self.zerocrit(zero_outputs, torch.zeros_like(zero_outputs))

        loss.backward()
        opt.second_step(zero_grad=True)

        self.scheduler.step()
        self.log('train_loss', loss, prog_bar=True)
        return None
    
    def validation_step(self, batch, batch_idx):
        vinputs, vlabels = batch["images"], batch["labels"]
        voutputs = self.model(vinputs)
        vloss = self.criterion(voutputs, vlabels)
        predicate_matrix = self.criterion.get_predicate_matrix()
        voutputs = self.criterion.binarize_output(voutputs)
        voutputs = voutputs.view(-1, 1, self.NUM_FEATURES)
        ANDed = voutputs * predicate_matrix
        diff = ANDed - voutputs
        voutputs = voutputs.view(-1, self.NUM_FEATURES)
        running_false_positives = ((predicate_matrix[vlabels] - voutputs) == -1).sum() / voutputs.shape[0]
        running_missing_attr = ((voutputs - predicate_matrix[vlabels]) == -1).sum() / voutputs.shape[0]
        running_out_attributes = voutputs.sum() / voutputs.shape[0]

        self.log_dict({
            'val_loss': vloss,
            'val_acc': self.accuracy(diff.sum(dim=2), vlabels),
            'val_fp': running_false_positives,
            'val_ma': running_missing_attr,
            'val_oa': running_out_attributes
            }, prog_bar=True)
        
        self.running_false_positives += running_false_positives
        self.running_missing_attr += running_missing_attr
        self.running_out_attributes += running_out_attributes
        self.step_count += 1

    def configure_optimizers(self):
        base_optimizer = torch.optim.Adam
        optimizer = SAM(self.parameters(), base_optimizer, lr=3e-5, weight_decay=1e-5)

        return optimizer
    
    def train_dataloader(self):
        return self.training_loader
    
    def val_dataloader(self):
        return self.validation_loader
    
# import torch datasets
from torch.utils.data import Dataset

class ZeroDataset(Dataset):
    def __init__(self, train_dataset, zero_dataset):
        self.train_dataset = train_dataset
        self.zero_dataset = zero_dataset

        # reduce the bigger dataset to the size of the smaller dataset
        is_train_bigger = len(train_dataset) > len(zero_dataset)

        if is_train_bigger:
            self.train_dataset = torch.utils.data.Subset(train_dataset, range(len(zero_dataset)))
        else:
            self.zero_dataset = torch.utils.data.Subset(zero_dataset, range(len(train_dataset)))

        self.length = len(self.train_dataset)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            "train_images": self.train_dataset[idx]["images"],
            "train_labels": self.train_dataset[idx]["labels"],
            "zero_images": self.zero_dataset[idx]["images"]
        }