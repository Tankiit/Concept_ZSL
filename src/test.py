import os

root='/notebooks/Concept_ZSL/results/AwA2-CAM'
classes = [d.name for d in os.scandir(root) if d.is_dir()]

feature_images = {}

for i, c in enumerate(classes):
    for f in os.listdir(os.path.join(root, c)):
        feat = f.split(".")[0]
        if feat not in feature_images:
            feature_images[feat] = [(os.path.join(root, c, f), c)]
        else:
            feature_images[feat].append((os.path.join(root, c, f), c))

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
make_dir(f"results/AwA2-CAMAttr")

import shutil
for key in feature_images:
    make_dir(f"results/AwA2-CAMAttr/{key}")
    for path, c in feature_images[key]:
        shutil.copy(path, f"results/AwA2-CAMAttr/{key}/{c}.jpg")