import clip, torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

from torchvision.datasets import CIFAR10
cifar100 = CIFAR10(root="./data", download=True, train=False)

image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)

attributes = ["feathers", "fur", "wings", "beak", "door", "ears", "paws", "mane", "tires", "metallic", "wheels", "claws", "whiskers", "antlers", "hooves", "eyes", "tail"]

text_inputs = torch.cat([clip.tokenize(c) for c in attributes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

print(image_features @ text_features.T)

similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{attributes[index]:>16s}: {100 * value.item():.2f}%")

import matplotlib.pyplot as plt

imgplot = plt.imshow(image)
plt.show()