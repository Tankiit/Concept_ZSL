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
from SubsetLoss import BSSLoss, pack_model
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
        
        self.running_false_positives += running_false_positives
        self.running_missing_attr += running_missing_attr
        self.running_out_attributes += running_out_attributes
        self.step_count += 1

    def configure_optimizers(self):
        base_optimizer = torch.optim.Adam
        optimizer = SAM(self.parameters(), base_optimizer, lr=3e-5, weight_decay=1e-5)

        return optimizer
    
    def train_dataloader(self):
        return training_loader
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
training_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

lightning_model = GZSLDeiT3()
trainer = L.Trainer(devices=1, max_epochs=EPOCHS, precision=16)
trainer.fit(lightning_model)

# Pack model
model = pack_model(lightning_model.model, lightning_model.criterion)

print("===============================================================")
print(f"Started On {NUM_EXCLUDE} Excluded Classes")

model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

avg_oa = lightning_model.running_out_attributes / lightning_model.step_count
avg_fp = lightning_model.running_false_positives / lightning_model.step_count
avg_ma = lightning_model.running_missing_attr / lightning_model.step_count
attributes_per_class = avg_oa - avg_fp + avg_ma

ZSL_test_loader = torch.utils.data.DataLoader(
        ZSL_valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
ZSL_training_loader = torch.utils.data.DataLoader(
        ZSL_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

accuracy = Accuracy(task="multiclass", num_classes=NUM_EXCLUDE, top_k=1).to(device)

predis = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)

with torch.no_grad():
    for i, vdata in enumerate(ZSL_training_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs = model(vinputs)

        for i in range(len(voutputs)):
            predis[vlabels[i]] += voutputs[i]
    
    K = int(attributes_per_class+1)
    print(f"K: {K}")
    topk, indices = torch.topk(predis, K, dim=1)

    new_predicate_matrix = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)
    new_predicate_matrix.scatter_(1, indices, 1)
    
    catted = torch.cat((model.predicate_matrix, new_predicate_matrix), dim=0)

    unseen_results = [[0,0]] * NUM_EXCLUDE
    for i, vdata in enumerate(ZSL_test_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs = model(vinputs)
        voutputs = voutputs.view(-1, 1, NUM_FEATURES)
        ANDed = voutputs * catted
        diff = ANDed - voutputs
        preds = diff.sum(dim=2)
        # Add to results[label] the number of correct predictions and the number of predictions
        for i in range(len(preds)):
            unseen_results[vlabels[i]][0] += torch.argmax(preds[i]) == vlabels[i]+150
            unseen_results[vlabels[i]][1] += 1
            
    seen_results = [[0,0]] * (NUM_CLASSES-NUM_EXCLUDE)
    for i, vdata in enumerate(valset):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs, _, _ = model(vinputs)
        voutputs = voutputs.view(-1, 1, NUM_FEATURES)
        ANDed = voutputs * catted
        diff = ANDed - voutputs
        preds = diff.sum(dim=2)
        for i in range(len(preds)):
            seen_results[vlabels[i]][0] += torch.argmax(preds[i]) == vlabels[i]
            seen_results[vlabels[i]][1] += 1

unseen_acc = 0
for i in range(NUM_EXCLUDE):
    unseen_acc += unseen_results[i][0] / unseen_results[i][1]
    
seen_acc = 0
for i in range(NUM_CLASSES-NUM_EXCLUDE):
    seen_acc += seen_results[i][0] / seen_results[i][1]
    
unseen_acc = unseen_acc / NUM_EXCLUDE
seen_acc = seen_acc / (NUM_CLASSES-NUM_EXCLUDE)

print(f"Unseen ACC: {unseen_acc}, Seen ACC: {seen_acc}, Harmonic ACC: {2*seen_acc*unseen_acc/(seen_acc+unseen_acc)}")