import torch
import numpy as np

NUM_FEATURES = 72
NUM_CLASSES = 196

import timm
model = timm.create_model("deit3_medium_patch16_224.fb_in22k_ft_in1k", pretrained=True).cuda()
model.head = torch.nn.Linear(512, NUM_FEATURES).cuda()
model.load_state_dict(torch.load("CarsDeiT3Auto.pt"))
model.eval()

from SubsetLoss import BSSLoss
loss_fn = BSSLoss(NUM_FEATURES, add_predicate_matrix=True, n_classes=NUM_CLASSES).cuda()
loss_fn.load_state_dict(torch.load("CarsDeiT3AutoLossFN.pt"))
loss_fn.eval()

print("#===============================================================")

import os

root='/storage/Cars/test_images'
classes = [d.name for d in os.scandir(root) if d.is_dir()]

images = []
for i, c in enumerate(classes):
    for f in os.listdir(os.path.join(root, c)):
        images += [os.path.join(root, c, f)]

import random
random.shuffle(images)
        
def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
print("#===============================================================")

from captum.attr import GuidedGradCam
guided_gc = GuidedGradCam(model, model.blocks[-1].norm1)

predicate_matrix = loss_fn.get_predicate_matrix()
print(predicate_matrix.shape)

import cv2
from torchvision.io.image import read_image, ImageReadMode
from torchvision.transforms.functional import normalize, resize, to_pil_image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from tqdm import tqdm
for i, image in enumerate(tqdm(images[:100])):
    c = classes.index(image.split("/")[-2])
    rgb_img = cv2.imread(image, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).cuda()

    voutput = model(input_tensor)

    make_dir(f"results/Cars-GCAM")
    make_dir(f"results/Cars-GCAM/img{i}")
    for j in range(NUM_FEATURES):
        if voutput[0][j] > 0.5:
            attribution = guided_gc.attribute(input_tensor, j)

            if predicate_matrix[c][j] == 1:
                cv2.imwrite(f"results/Cars-GCAM/img{i}/feature{j}.jpg", attribution)
            else:
                cv2.imwrite(f"results/Cars-GCAM/img{i}/NOTfeature{j}.jpg", attribution)