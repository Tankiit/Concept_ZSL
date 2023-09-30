from torchvision.io.image import read_image, ImageReadMode
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp

print("1========================================================================")

import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-1]) + "/models")
from ResnetAutoPredicates import ResExtr

NUM_FEATURES = 64
NUM_CLASSES = 200

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResExtr(NUM_FEATURES, NUM_CLASSES, resnet_type=18, pretrained=True).to(device)
model.load_state_dict(torch.load("CUBRes18AutoPred.pt"))
model.eval()

print("2========================================================================")

import pandas as pd
import os

root = "/storage/CUB"

images = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
image_class_labels = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
data = images.merge(image_class_labels, on='img_id')
train_test_split = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'train_test_split.txt'),
                               sep=' ', names=['img_id', 'is_training_img'])

data = data.merge(train_test_split, on='img_id')

data = data[data.is_training_img == 0]
images = data["filepath"].tolist()

from torchcam.utils import overlay_mask

from tqdm import tqdm

print("3========================================================================")

import matplotlib.pyplot as plt

cam_extractor = SmoothGradCAMpp(model.resnet, ["layer1", "layer2", "layer3", "layer4"])
for i, image in tqdm(enumerate(images[:200])):
    img = read_image(os.path.join(root, 'CUB_200_2011/images', image), ImageReadMode.RGB)
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
    voutputs, _, _ = model(input_tensor.unsqueeze(0))
    
    os.mkdir(f"results/CUB-CAM/img{i}")
    for j in range(NUM_FEATURES):
        if voutputs[0][j] == 1:
            activation_map = cam_extractor(j, voutputs)
            
            os.mkdir(f"results/CUB-CAM/img{i}/feature{j}")
            
            # The raw CAM
            for idx, name, cam in zip(range(len(cam_extractor.target_names)), cam_extractor.target_names, activation_map):
                result = overlay_mask(to_pil_image(img), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)
                result.save(f"results/CUB-CAM/img{i}/feature{j}/layer-{name}.jpg")