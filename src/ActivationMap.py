from torchvision.io.image import read_image, ImageReadMode
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import LayerCAM

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
images = pd.read_csv(os.path.join("/storage/CUB", 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
image_class_labels = pd.read_csv(os.path.join("/storage/CUB", 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
data = images.merge(image_class_labels, on='img_id')
train_test_split = pd.read_csv(os.path.join("/storage/CUB", 'CUB_200_2011', 'train_test_split.txt'),
                               sep=' ', names=['img_id', 'is_training_img'])

data = data.merge(train_test_split, on='img_id')

data = data[data.is_training_img == 0]
images = data["filepath"].tolist()

from torchcam.utils import overlay_mask

from tqdm import tqdm

print("3========================================================================")

import matplotlib.pyplot as plt

for i, image in tqdm(enumerate(images[:500])):
    img = read_image(os.path.join("/storage/CUB", 'CUB_200_2011/images', image), ImageReadMode.RGB)
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
    # Preprocess your data and feed it to the model
    with torch.no_grad():
        outs = model.resnet(input_tensor.unsqueeze(0))
        
    activateds = [j for j, out in enumerate(outs[0]) if out > 0.5]
    
    # Retrieve the CAM by passing the class index and the model output
    for j in activateds:
        cam_extractor = LayerCAM(model.resnet, ["layer1", "layer2", "layer3", "layer4"])
        out = model.resnet(input_tensor.unsqueeze(0))
        cams = cam_extractor(j, out)

        # The raw CAM
        _, axes = plt.subplots(1, len(cam_extractor.target_names) + 1)
        for idx, name, cam in zip(range(len(cam_extractor.target_names)), cam_extractor.target_names, cams):
            axes[idx].imshow(cam.squeeze(0).cpu().numpy()); axes[idx].axis('off'); axes[idx].set_title(name);

        axes[-1].imshow(img.reshape(img.shape[1], img.shape[2], img.shape[0]).numpy()); axes[-1].axis('off'); axes[-1].set_title("Base Image");

        plt.savefig(f"results/CUB-CAM/img{i}-feature{j}.png")
        plt.close()
        
        cam_extractor.remove_hooks()

        #result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        #result.save(f"results/CUB-CAM/img{i}-feature{j}.jpg")
