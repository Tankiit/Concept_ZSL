import torch
    
CLASSES = 10
NUM_FEATURES = 5
OUTS = 2
    
predicate_matrix = torch.randint(0, 2, (CLASSES, NUM_FEATURES))
print(predicate_matrix)

avg_attr = predicate_matrix.sum(dim=1)
print(avg_attr)