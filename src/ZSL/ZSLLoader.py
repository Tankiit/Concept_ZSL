import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import torch
from PIL import Image
import os

DATASET = 'CUB' # ["AWA2", "CUB", "SUN"]

if DATASET == 'AWA2':
    ROOT='./data/AWA2/Animals_with_Attributes2/JPEGImages/'
elif DATASET == 'CUB':
    ROOT='/storage/CUB/CUB_200_2011/images/'
elif DATASET == 'SUN':
    ROOT='./data/SUN/images/'
else:
    print("Please specify the dataset")
    
DATA_DIR = f'/storage/xlsa17/data/{DATASET}'
data = sio.loadmat(f'{DATA_DIR}/res101.mat')
# data consists of files names 
attrs_mat = sio.loadmat(f'{DATA_DIR}/att_splits.mat')
# attrs_mat is the attributes (class-level information)
image_files = data['image_files']

if DATASET == 'AWA2':
    image_files = np.array([im_f[0][0].split('JPEGImages/')[-1] for im_f in image_files])
else:
    image_files = np.array([im_f[0][0].split('images/')[-1] for im_f in image_files])

# labels are indexed from 1 as it was done in Matlab, so 1 subtracted for Python
labels = data['labels'].squeeze().astype(np.int64) - 1
train_idx = attrs_mat['train_loc'].squeeze() - 1
val_idx = attrs_mat['val_loc'].squeeze() - 1
trainval_idx = attrs_mat['trainval_loc'].squeeze() - 1
test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1

# consider the train_labels and val_labels
train_labels = labels[train_idx]
val_labels = labels[val_idx]

# split train_idx to train_idx (used for training) and val_seen_idx
train_idx, val_seen_idx = train_test_split(train_idx, test_size=0.2, stratify=train_labels)
# split val_idx to val_idx (not useful) and val_unseen_idx
val_unseen_idx = train_test_split(val_idx, test_size=0.2, stratify=val_labels)[1]

### used for validation
# train files and labels
train_files = image_files[train_idx]
train_labels = labels[train_idx]
uniq_train_labels, train_labels_based0, counts_train_labels = np.unique(train_labels, return_inverse=True, return_counts=True)
# val seen files and labels
val_seen_files = image_files[val_seen_idx]
val_seen_labels = labels[val_seen_idx]
uniq_val_seen_labels = np.unique(val_seen_labels)
# val unseen files and labels
val_unseen_files = image_files[val_unseen_idx]
val_unseen_labels = labels[val_unseen_idx]
uniq_val_unseen_labels = np.unique(val_unseen_labels)

### used for testing
# trainval files and labels
trainval_files = image_files[trainval_idx]
trainval_labels = labels[trainval_idx]
uniq_trainval_labels, trainval_labels_based0, counts_trainval_labels = np.unique(trainval_labels, return_inverse=True, return_counts=True)
# test seen files and labels
test_seen_files = image_files[test_seen_idx]
test_seen_labels = labels[test_seen_idx]
uniq_test_seen_labels = np.unique(test_seen_labels)
# test unseen files and labels
test_unseen_files = image_files[test_unseen_idx]
test_unseen_labels = labels[test_unseen_idx]
uniq_test_unseen_labels = np.unique(test_unseen_labels)

class DataLoader(Dataset):
    def __init__(self, root, image_files, labels, transform=None):
        self.root  = root
        self.image_files = image_files
        self.labels = labels 
        self.transform = transform

    def __getitem__(self, idx):
        # read the iterable image
        img_pil = Image.open(os.path.join(self.root, self.image_files[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img_pil)
        # label
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.image_files)

trainTransform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
# Testing Transformations
testTransform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def train(model, data_loader, optimizer, scheduler):
    loss_meter = AverageMeter()
    """ train the model  """
    model.train()
    tk = tqdm(data_loader, total=int(len(data_loader)))
    for batch_idx, (inputs, labels) in enumerate(tk):
        inputs, labels = inputs.cuda(), labels.cuda()
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs, commit_loss, predicate_matrix = model(inputs)

        # Compute the loss and its gradients
        if torch.any(torch.isnan(model.predicate_matrix)):
            exit()
        loss = loss_fn(outputs, labels, predicate_matrix) + commit_loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        loss_meter.update(loss.item(), labels.shape[0])
        tk.set_postfix({"loss": loss_meter.avg})
    
    # print training/validation statistics 
    print('Train: Average loss: {:.4f}\n'.format(loss_meter.avg))

eps=1e-10
def loss_fn(out, labels, predicate_matrix):
    out = out.view(-1, 1, NUM_FEATURES) # out is a batch of 1D binary vectors
    ANDed = out * predicate_matrix # AND operation
    diff = ANDed - out # Difference of ANDed and out => if equal, then out is a subset of its class' predicates

    entr_loss = torch.nn.CrossEntropyLoss()
    loss_cl = entr_loss(diff.sum(dim=2), labels) # Is "out" a subset of its class' predicates?

    batch_size = out.shape[0]

    out = out.view(-1, NUM_FEATURES)
    diff_square = (out - predicate_matrix[labels]).pow(2)
    
    false_positives = (out - predicate_matrix[labels] + diff_square).sum() / batch_size
    missing_attr = (predicate_matrix[labels] - out + diff_square).sum() / batch_size
    
    loss_ft = (1 + false_positives + missing_attr)
    loss_ft *= loss_cl.item()/(loss_ft.item() + eps)
    
    return loss_cl + loss_ft * FT_WEIGHT

def get_reprs(model, data_loader):
    model.eval()
    reprs = []
    for _, (vinputs, _) in enumerate(data_loader):
        vinputs = vinputs.cuda()
        
        with torch.no_grad():
            voutputs, _, predicate_matrix = model(vinputs)
            
            voutputs = voutputs.view(-1, 1, NUM_FEATURES)
            ANDed = voutputs * predicate_matrix
            diff = ANDed - voutputs
            feat = diff.sum(dim=2)
            
        reprs.append(feat.cpu().data.numpy())
        
    reprs = np.concatenate(reprs, 0)
    return reprs

def compute_accuracy(pred_labels, true_labels, labels):
    acc_per_class = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        idx = (true_labels == labels[i])
        acc_per_class[i] = np.sum(pred_labels[idx] == true_labels[idx]) / np.sum(idx)
    return np.mean(acc_per_class)

def validation(model, seen_loader, seen_labels, unseen_loader, unseen_labels, attrs_mat):
    # Representation
    with torch.no_grad():
        seen_reprs = get_reprs(model, seen_loader)
        unseen_reprs = get_reprs(model, unseen_loader)

    # Labels
    uniq_labels = np.unique(np.concatenate([seen_labels, unseen_labels]))
    updated_seen_labels = np.searchsorted(uniq_labels, seen_labels)
    uniq_updated_seen_labels = np.unique(updated_seen_labels)
    updated_unseen_labels = np.searchsorted(uniq_labels, unseen_labels)
    uniq_updated_unseen_labels = np.unique(updated_unseen_labels)
    uniq_updated_labels = np.unique(np.concatenate([updated_seen_labels, updated_unseen_labels]))

    # truncate the attribute matrix
    trunc_attrs_mat = attrs_mat[uniq_labels]
    
    #### ZSL ####
    pred_labels = np.argmax(unseen_reprs, axis=1)
    zsl_unseen_predict_labels = uniq_updated_unseen_labels[pred_labels]
    zsl_unseen_acc = compute_accuracy(zsl_unseen_predict_labels, updated_unseen_labels, uniq_updated_unseen_labels)
    
    #### GZSL ####
    gzsl_seen_pred_labels = np.argmax(seen_reprs, axis=1)
    # gzsl_seen_predict_labels = uniq_updated_labels[pred_seen_labels]
    gzsl_seen_acc = compute_accuracy(gzsl_seen_pred_labels, updated_seen_labels, uniq_updated_seen_labels)

    gzsl_unseen_pred_labels = np.argmax(unseen_reprs, axis=1)
    # gzsl_unseen_predict_labels = uniq_updated_labels[pred_unseen_labels]
    gzsl_unseen_acc = compute_accuracy(gzsl_unseen_pred_labels, updated_unseen_labels, uniq_updated_unseen_labels)

    H = 2 * gzsl_seen_acc * gzsl_unseen_acc / (gzsl_seen_acc + gzsl_unseen_acc)

    print('ZSL: averaged per-class accuracy: {0:.2f}'.format(zsl_unseen_acc * 100))
    print('GZSL Seen: averaged per-class accuracy: {0:.2f}'.format(gzsl_seen_acc * 100))
    print('GZSL Unseen: averaged per-class accuracy: {0:.2f}'.format(gzsl_unseen_acc * 100))
    print('GZSL: harmonic mean (H): {0:.2f}'.format(H * 100))
    
def test(model, test_seen_loader, test_seen_labels, test_unseen_loader, test_unseen_labels):
    # Representation
    with torch.no_grad():
        seen_reprs = get_reprs(model, test_seen_loader)
        unseen_reprs = get_reprs(model, test_unseen_loader)
    # Labels
    uniq_test_seen_labels = np.unique(test_seen_labels)
    uniq_test_unseen_labels = np.unique(test_unseen_labels)

    # ZSL
    predict_labels = np.argmax(unseen_reprs, axis=1)
    zsl_unseen_predict_labels = uniq_test_unseen_labels[predict_labels]
    zsl_unseen_acc = compute_accuracy(zsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)

    # Calibrated stacking
    Cs_mat = np.zeros(attrs_mat.shape[0])
    Cs_mat[uniq_test_seen_labels] = gamma

    # GZSL
    # seen classes
    gzsl_seen_predict_labels = np.argmax(seen_reprs, axis=1)
    gzsl_seen_acc = compute_accuracy(gzsl_seen_predict_labels, test_seen_labels, uniq_test_seen_labels)
    
    # unseen classes
    gzsl_unseen_predict_labels = np.argmax(unseen_reprs, axis=1)
    gzsl_unseen_acc = compute_accuracy(gzsl_unseen_predict_labels, test_unseen_labels, uniq_test_unseen_labels)

    H = 2 * gzsl_unseen_acc * gzsl_seen_acc / (gzsl_unseen_acc + gzsl_seen_acc)

    print('ZSL: averaged per-class accuracy: {0:.2f}'.format(zsl_unseen_acc * 100))
    print('GZSL Seen: averaged per-class accuracy: {0:.2f}'.format(gzsl_seen_acc * 100))
    print('GZSL Unseen: averaged per-class accuracy: {0:.2f}'.format(gzsl_unseen_acc * 100))
    print('GZSL: harmonic mean (H): {0:.2f}'.format(H * 100))
    print('GZSL: gamma: {0:.2f}'.format(gamma))
    
num_workers = 4
### used in validation
# train data loader
train_data = DataLoader(ROOT, train_files, train_labels_based0, transform=trainTransform)
weights_ = 1. / counts_train_labels
weights = weights_[train_labels_based0]
train_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=train_labels_based0.shape[0], replacement=True)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, sampler=train_sampler, num_workers=num_workers)
# seen val data loader
val_seen_data = DataLoader(ROOT, val_seen_files, val_seen_labels, transform=testTransform)
val_seen_data_loader = torch.utils.data.DataLoader(val_seen_data, batch_size=256, shuffle=False, num_workers=num_workers)
# unseen val data loader
val_unseen_data = DataLoader(ROOT, val_unseen_files, val_unseen_labels, transform=testTransform)
val_unseen_data_loader = torch.utils.data.DataLoader(val_unseen_data, batch_size=256, shuffle=False, num_workers=num_workers)

### used in testing
# trainval data loader
trainval_data = DataLoader(ROOT, trainval_files, trainval_labels_based0, transform=trainTransform)
weights_ = 1. / counts_trainval_labels
weights = weights_[trainval_labels_based0]
trainval_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=trainval_labels_based0.shape[0], replacement=True)
trainval_data_loader = torch.utils.data.DataLoader(trainval_data, batch_size=32, sampler=trainval_sampler, num_workers=num_workers)
# seen test data loader
test_seen_data = DataLoader(ROOT, test_seen_files, test_seen_labels, transform=testTransform)
test_seen_data_loader = torch.utils.data.DataLoader(test_seen_data, batch_size=256, shuffle=False, num_workers=num_workers)
# unseen test data loader
test_unseen_data = DataLoader(ROOT, test_unseen_files, test_unseen_labels, transform=testTransform)
test_unseen_data_loader = torch.utils.data.DataLoader(test_unseen_data, batch_size=256, shuffle=False, num_workers=num_workers)

import collections
from torch import optim

NUM_CLASSES = 200
NUM_FEATURES = 64
EPOCHS = 50

FT_WEIGHT = 1

import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/models")
from ResnetAutoPredicates import ResExtr

model = ResExtr(NUM_FEATURES, NUM_CLASSES, resnet_type=18, pretrained=True).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 2e-3, epochs=EPOCHS, steps_per_epoch=len(training_loader))
scheduler = None

for i in range(EPOCHS):
    train(model, trainval_data_loader, optimizer, scheduler)
    print('Epoch: ', i)
    test(model, test_seen_data_loader, test_seen_labels, test_unseen_data_loader, test_unseen_labels)