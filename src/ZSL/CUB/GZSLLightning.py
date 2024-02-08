import torchvision.transforms as transforms
train_transform =  transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

NUM_CLASSES = 200
NUM_FEATURES = 64
EPOCHS = 50
NUM_EXCLUDE = 50
BATCH_SIZE = 64

import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/train/CUB")
from CUBLoader import make_ZSL_sets
trainset, valset, ZSL_trainset, ZSL_valset = make_ZSL_sets("datasets/", NUM_EXCLUDE, train_transform, val_transform)

import torch
torch.set_float32_matmul_precision('medium')

import lightning as L
import timm
sys.path.insert(0, "/".join(__file__.split("/")[:-3]))
from SubsetLoss import BSSLoss
sys.path.insert(0, "/".join(__file__.split("/")[:-2]))
from sam import SAM
from torchmetrics import Accuracy
class GZSLDeiT3(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("deit3_medium_patch16_224.fb_in22k_ft_in1k", pretrained=True)
        self.model.head = torch.nn.Linear(512, NUM_FEATURES)
        self.criterion = BSSLoss(NUM_FEATURES, add_predicate_matrix=True, n_classes=NUM_CLASSES - NUM_EXCLUDE)
        self.accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES - NUM_EXCLUDE, top_k=1)

        optimizer = self.configure_optimizers()
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer.base_optimizer, 1e-4, epochs=EPOCHS, steps_per_epoch=len(training_loader))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch["images"], batch["labels"]
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        opt = self.optimizers()
        
        loss.backward()
        opt.first_step(zero_grad=True)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        
        loss.backward()
        opt.second_step(zero_grad=True)

        self.scheduler.step()
        self.log('train_loss', loss)
        return None
    
    def validation_step(self, batch, batch_idx):
        vinputs, vlabels = batch["images"], batch["labels"]
        voutputs = model(vinputs)
        vloss = self.criterion(voutputs, vlabels)
        predicate_matrix = self.criterion.get_predicate_matrix()
        voutputs = self.criterion.binarize_output(voutputs)
        voutputs = voutputs.view(-1, 1, NUM_FEATURES)
        ANDed = voutputs * predicate_matrix
        diff = ANDed - voutputs
        voutputs = voutputs.view(-1, NUM_FEATURES)
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

    def configure_optimizers(self):
        base_optimizer = torch.optim.Adam
        optimizer = SAM(self.parameters(), base_optimizer, lr=3e-5, weight_decay=1e-5)

        return optimizer
    
    def train_dataloader(self):
        return training_loader
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
training_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = GZSLDeiT3()
trainer = L.Trainer(devices=1, max_epochs=EPOCHS, precision=16)
trainer.fit(model)