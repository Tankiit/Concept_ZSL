import torch
import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/models")
from VectorAutoPredicates import ResExtr
from torchmetrics import Accuracy

def train_one_epoch(model, optimizer, NUM_FEATURES, FT_WEIGHT, POS_FT_WEIGHT):
    running_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data['features'].to(device), data['labels'].to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs, commit_loss, predicate_matrix = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels, predicate_matrix, NUM_FEATURES, FT_WEIGHT, POS_FT_WEIGHT) + commit_loss
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i+1)

eps=1e-10
def loss_fn(out, labels, predicate_matrix, NUM_FEATURES, FT_WEIGHT, POS_FT_WEIGHT):
    out = out.view(-1, 1, NUM_FEATURES) # out is a batch of 1D binary vectors
    ANDed = out * predicate_matrix # AND operation
    diff = ANDed - out # Difference of ANDed and out => if equal, then out is a subset of its class' predicates

    entr_loss = torch.nn.CrossEntropyLoss()
    loss_cl = entr_loss(diff.sum(dim=2), labels) # Is "out" a subset of its class' predicates?

    batch_size = out.shape[0]

    classes = torch.zeros(batch_size, NUM_CLASSES, device="cuda")
    classes[torch.arange(batch_size), labels] = 1
    classes = classes.view(batch_size, NUM_CLASSES, 1).expand(batch_size, NUM_CLASSES, NUM_FEATURES)

    extra_features = out - predicate_matrix + (out - predicate_matrix).pow(2)

    loss_neg_ft = torch.masked_select(extra_features, (1-classes).bool()).view(-1, NUM_FEATURES).sum() / batch_size

    labels_predicate = predicate_matrix[labels]
    extra_features_in = torch.masked_select(extra_features, classes.bool()).view(-1, NUM_FEATURES)
    loss_pos_ft = (labels_predicate - out.view(batch_size, NUM_FEATURES) + extra_features_in/2).sum() / batch_size

    return loss_cl + loss_neg_ft * FT_WEIGHT * loss_cl.item()/(loss_neg_ft.item() + eps) + loss_pos_ft * POS_FT_WEIGHT * loss_cl.item()/(loss_pos_ft.item() + eps)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import optim
def objective(ga_instance, solution, solution_idx):
    global trial_num
    trial_num += 1
    print(f"Starting trial {trial_num}")
    # Generate the model.

    NUM_FEATURES = min(max(round(solution[0]), 1), 8)
    FT_WEIGHT = min(max(solution[1], 0), 2)
    POS_FT_WEIGHT = min(max(solution[2], 0), 2)
    lr = min(max(solution[3], 1e-5), 1e-2)

    model = ResExtr(2048, NUM_FEATURES*16, NUM_CLASSES).to(device)

    # Generate the optimizers.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    EPOCHS = 30

    best_acc = 0.0

    for epoch in range(EPOCHS):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        _ = train_one_epoch(model, optimizer, NUM_FEATURES*16, FT_WEIGHT, POS_FT_WEIGHT)

        model.eval()
        running_acc = 0.0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata['features'].to(device), vdata['labels'].to(device)
                voutputs, _, predicate_matrix = model(vinputs)
                voutputs = voutputs.view(-1, 1, NUM_FEATURES*16)
                ANDed = voutputs * predicate_matrix
                diff = ANDed - voutputs
                running_acc += accuracy(diff.sum(dim=2), vlabels)

        avg_acc = running_acc / (i + 1)

        if best_acc < avg_acc:
            best_acc = avg_acc

        if epoch > 2 and best_acc < 0.5:
            return 0
    
    print(f"Trial {trial_num} finished with accuracy {best_acc}, parameters {solution}")
    return best_acc

if __name__ == "__main__":
    NUM_CLASSES = 50

    import pickle

    # pickle val_set
    with open("val_set.pkl", "rb") as f:
        val_set = pickle.load(f)

    # pickle train_set
    with open("train_set.pkl", "rb") as f:
        train_set = pickle.load(f)

    training_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)

    accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)
    trial_num = -1
    fitness_function = objective

    num_generations = 50
    num_parents_mating = 4

    sol_per_pop = 8
    num_genes = 4

    init_range_low = 0
    init_range_high = 5

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 25

    import pygad
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        init_range_low=init_range_low,
                        init_range_high=init_range_high,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes)

    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
