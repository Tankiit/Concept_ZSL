from torchvision.io.image import read_image, ImageReadMode
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp

print("1========================================================================")

import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-1]) + "/models")
from DeiT3AutoPredicates import ResExtr

NUM_FEATURES = 80
NUM_CLASSES = 196

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResExtr(NUM_FEATURES, NUM_CLASSES, pretrained=True).to(device)
model.load_state_dict(torch.load("CarsDeiT3.pt"))
model.eval()

print(model)

print("2========================================================================")

import torchvision.transforms.v2 as transforms
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

import os

root='/storage/Cars/test_images'
classes = [d.name for d in os.scandir(root) if d.is_dir()]

images = []
for i, c in enumerate(classes):
    for f in os.listdir(os.path.join(root, c)):
        images += [os.path.join(root, c, f)]
    
from torchcam.utils import overlay_mask

from tqdm import tqdm

print("3========================================================================")

import matplotlib.pyplot as plt

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

cam_extractor = SmoothGradCAMpp(model.deit.blocks[0], ["norm1"])
for i, image in tqdm(enumerate(images[:200])):
    img = read_image(image, ImageReadMode.RGB)
    input_tensor = val_transform(img.float()).to(device)
    voutputs, _, _ = model(input_tensor.unsqueeze(0))
    
    make_dir(f"results/Cars-CAM/img{i}")
    for j in range(NUM_FEATURES):
        if voutputs[0][j] == 1:
            activation_map = cam_extractor(j, voutputs)
            
            make_dir(f"results/Cars-CAM/img{i}/feature{j}")
            
            # The raw CAM
            for idx, name, cam in zip(range(len(cam_extractor.target_names)), cam_extractor.target_names, activation_map):
                print(input_tensor.shape, cam.shape)
                result = overlay_mask(to_pil_image(input_tensor), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)
                result.save(f"results/Cars-CAM/img{i}/feature{j}/layer-{name}.jpg")