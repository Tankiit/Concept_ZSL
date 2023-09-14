from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

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

image_root = "data/CUB_200_2011"

images = pd.read_csv(os.path.join(image_root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
image_class_labels = pd.read_csv(os.path.join(image_root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
data = images.merge(image_class_labels, on='img_id')
train_test_split = pd.read_csv(os.path.join(image_root, 'CUB_200_2011', 'train_test_split.txt'),
                               sep=' ', names=['img_id', 'is_training_img'])

data = data.merge(train_test_split, on='img_id')

data = data[data.is_training_img == 0]
images = data["filepath"].tolist()

import random
random.shuffle(images)

images = images[:250]

# sort alphabetically
images.sort()

from tqdm import tqdm

print("3========================================================================")

for i, image in tqdm(enumerate(images)):
    img = read_image(os.path.join(image_root, 'CUB_200_2011/images', image))
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
    voutputs, _, _ = model(input_tensor.unsqueeze(0))
    
    # Write to file
    with open(f"results/CUB-IMGAttr/image-{i}.txt", "w") as text_file:
        text_file.write(f"{image}\n")
        for j in range(NUM_FEATURES):
            text_file.write(f"{voutputs[0][j].item()}\n")
        text_file.write("\n")