# get all attributes

print("Getting attributes...")

attributes_root = "results/CUB-Attributes"

attributes = set()

import os

for filename in os.listdir(attributes_root):
    with open(os.path.join(attributes_root, filename)) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("- "):
                attribute = line[2:].strip()
                attributes.add(attribute)

attributes = list(attributes)

# ==================================================================================================
# get image paths

print("Getting images...")

IMGAttr_root = "results/CUB-IMGAttr"

images = []

for filename in os.listdir(IMGAttr_root):
    with open(os.path.join(IMGAttr_root, filename)) as f:
        lines = f.readlines()
        image = lines[0].strip()
        images.append(image)

# ==================================================================================================
# compare captions to images using CLIP

from tqdm import tqdm
from PIL import Image
import torch

TOPK = 100

import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

print("Encoding captions...")

MAX_CAPTIONS = 3000
CAPTION_BUCKETS = len(attributes) // MAX_CAPTIONS + 1

image_root = "data/CUB_200_2011"

images.sort()

print("Comparing images to captions...")
for i, image_path in tqdm(enumerate(images)):
    pil_image = Image.open(os.path.join(image_root, 'CUB_200_2011/images', image_path))
    image = preprocess(pil_image).unsqueeze(0).to(device)

    all_probs = []
    with torch.no_grad():
        for j in range(CAPTION_BUCKETS):
            text = clip.tokenize(attributes[j*MAX_CAPTIONS:(j+1)*MAX_CAPTIONS]).to(device)
            logits_per_image, logits_per_text = model(image, text)
            all_probs.append(logits_per_image)

    # get topk captions
    probs = torch.cat(all_probs, dim=1).softmax(dim=-1).cpu().numpy()
    topk = probs[0].argsort()[-TOPK:][::-1]
    topk_captions = [attributes[x] for x in topk]

    # save to file
    with open(f"results/CUB-IMGAttr/image-{i}.txt", "a") as f:
        f.write(f"{image_path}\n")
        for caption in topk_captions:
            f.write(f"{caption}\n")