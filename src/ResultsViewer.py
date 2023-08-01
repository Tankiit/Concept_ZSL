from AwA2ResnetExtLightning import ResnetExtractorLightning
model = ResnetExtractorLightning.load_from_checkpoint("lightning_logs/version_4/checkpoints/epoch=1-step=934.ckpt")

# disable randomness, dropout, etc...
model.eval()

# load the test set
import pickle, torch
with open("val_set.pkl", "rb") as f:
    val_set = pickle.load(f)

# get random samples
indices = torch.randint(0, len(val_set), (10,))
samples = [val_set[i] for i in indices]

# get the features and labels
features = torch.stack([sample["features"] for sample in samples])
labels = torch.stack([sample["labels"] for sample in samples])

from numpy import loadtxt
predicate_matrix_file = "datasets/Animals_with_Attributes2/predicate-matrix-binary.txt"
predicate_matrix = torch.from_numpy(loadtxt(predicate_matrix_file, dtype=int)).cuda()

file_paths = val_set.dataset.file_paths
dataset_files = [file_paths[i] for i in val_set.indices]
taken_file_paths = [dataset_files[i] for i in indices]

predicates = loadtxt('datasets/Animals_with_Attributes2/predicates.txt', dtype=str)
classes = loadtxt('datasets/Animals_with_Attributes2/classes.txt', dtype=str)

feature_mapping = {int(i)-1: feature for i, feature in predicates}
class_mapping = {int(i)-1: class_ for i, class_ in classes}
print("======================================================================")
print(feature_mapping)
print(class_mapping)
print("======================================================================")

# get the predictions
with torch.no_grad():
    predictions, _ = model(features.cuda())

    outputs = predictions.view(-1, 1, 85)
    ANDed = outputs * predicate_matrix
    diff = ANDed - outputs
    predicted_labels = diff.sum(dim=2).argmax(dim=1)

save_dir = "results/"
import os
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# print the results
for i in range(len(samples)):
    print("======================================================================")
    print("Predicted: {}, Actual: {}".format(predicted_labels[i], labels[i]))
    
    # print class predicates
    print("Class predicates:")
    print(predicate_matrix[labels[i]])
    print("Predicted predicates:")
    print(outputs[i])

    false_positives = ((predicate_matrix[labels[i]] - outputs[i]) == -1).sum()
    false_negatives = ((predicate_matrix[labels[i]] - outputs[i]) == 1).sum()
    print("Errors:", 85 - (outputs[i] == predicate_matrix[labels[i]]).sum().item(), "False positives:", false_positives.item(), "False negatives:", false_negatives.item())

    # Save file of images, predicted labels, and predicted features
    with open(save_dir + f"Sample_{i}.txt", "w") as f:
        f.write(f"Predicted: {predicted_labels[i]+1}, Actual: {labels[i]+1}\n")
        f.write("Class predicates:\n")
        f.write(str(predicate_matrix[labels[i]].cpu().numpy().tolist()) + "\n")
        f.write("Predicted predicates:\n")
        f.write(str(outputs[i][0].cpu().numpy().tolist()) + "\n")
        f.write(f"Errors: {85 - (outputs[i] == predicate_matrix[labels[i]]).sum().item()}, False positives: {false_positives.item()}, False negatives: {false_negatives.item()}\n")
        f.write(f"File path: {taken_file_paths[i]}\n")
        f.write(f"Predicted class: {class_mapping[predicted_labels[i].item()]}\n")
        f.write(f"Actual class: {class_mapping[labels[i].item()]}\n")
        f.write(f"Predicted features: {', '.join([feature_mapping[int(i)] for i, feature in enumerate(outputs[i][0]) if feature.item() == 1])}\n")
        f.write(f"Actual features: {', '.join([feature_mapping[int(i)] for i, feature in enumerate(predicate_matrix[labels[i]]) if feature.item() == 1])}\n")