import torch

eps=1e-10
def loss_fn(out, labels, predicate_matrix):
    out = out.view(-1, 1, NUM_FEATURES) # out is a batch of 1D binary vectors
    ANDed = out * predicate_matrix # AND operation
    diff = ANDed - out # Difference of ANDed and out => if equal, then out is a subset of its class' predicates
    print(diff.sum(dim=2))
    
    out = out.view(-1, NUM_FEATURES)
    diff_square = (out - predicate_matrix[labels]).pow(2)
    
    missing_attr = (predicate_matrix[labels] - out + diff_square)
    print(missing_attr)
    
CLASSES = 10
NUM_FEATURES = 5
OUTS = 2
    
outs = torch.randint(0, 2, (OUTS, NUM_FEATURES))
labels = torch.randint(0, CLASSES, (OUTS,))
predicate_matrix = torch.randint(0, 2, (CLASSES, NUM_FEATURES))

loss_fn(outs, labels, predicate_matrix)