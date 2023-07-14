class Tree:
    def __init__(self, root):
        self.root = root

    def forward(self, x):
        # x is a list of feature ids
        return self.root.forward(x)

    def build_network(self, feature_vector):
        """
        Calculate required network size, and build the network
        Train the network until 100% accuracy
        Make custom loss function to maximize value on class index and set all other values to 0 (can also try with using cross entropy loss)
        """
        pass
    
    def prnt(self):
        self.root.prnt(0)

class Node:
    def __init__(self, feature_id, left, right, _class=None):
        self.feature = feature_id
        self.left = left
        self.right = right
        self._class = _class

    def forward(self, x):
        if self.is_leaf():
            return self._class
        if self.feature in x:
            return self.left.forward(x)
        else:
            return self.right.forward(x)
        
    def is_leaf(self):
        return self._class != None
    
    def __repr__(self) -> str:
        if self.is_leaf():
            return f"Leaf: {self._class}"
        else:
            return f"Node: {self.feature}"

    def prnt(self, depth):
        if self.is_leaf():
            print("-" * depth + f"Leaf: {self._class}")
        else:
            print("-" * depth + f"Node: {self.feature}")
            self.left.prnt(depth + 1)
            self.right.prnt(depth + 1)

from uuid import uuid4
def recursive_build(feature_bank):
    # feature_bank is a dictionary of uuid: (class, feature set)

    # get the most common feature
    feature_count = {}
    for _, feature_set in feature_bank.values():
        for feature in feature_set:
            if feature not in feature_count:
                feature_count[feature] = 0
            feature_count[feature] += 1

    most_common_feature = max(feature_count, key=feature_count.get)

    root = Node(most_common_feature, None, None)

    # split the feature bank into two
    left_feature_bank = {}
    right_feature_bank = {}

    left_class = None
    right_class = None

    for _, (_class, feature_set) in feature_bank.items():
        if most_common_feature in feature_set:
            new_feature_set = feature_set.copy()
            new_feature_set.remove(most_common_feature)
            left_feature_bank[uuid4()] = (_class, new_feature_set)
            left_class = _class
        else:
            right_feature_bank[uuid4()] = (_class, feature_set)
            right_class = _class

    # build the left and right subtrees
    if len(left_feature_bank) == 1:
        left = Node(None, None, None, left_class)
    else:
        left = recursive_build(left_feature_bank)

    if len(right_feature_bank) == 1:
        right = Node(None, None, None, right_class)
    else:
        right = recursive_build(right_feature_bank)

    root.left = left
    root.right = right

    return root

def build_tree(feature_bank):
    # feature_bank is a dictionary of class: feature list, feature list is a list of feature sets
    # => flatten the dictionary into {"random_id": (class, feature_set)}
    feature_bank = {uuid4(): (_class, feature_set) for _class, feature_list in feature_bank.items() for feature_set in feature_list}

    return recursive_build(feature_bank)

def extand1_feature_bank(feature_bank):
    new_feature_bank = {}
    for _class, feature_list in feature_bank.items():
        new_feature_bank[_class] = feature_list.copy()

    for i, (_class, feature_list) in enumerate(feature_bank.items()):
        for feature_set in feature_list:
            for feature in feature_set:
                new_feature_set = feature_set.copy()
                new_feature_set.remove(feature)
                if new_feature_set in new_feature_bank[_class]:
                    continue
                to_add = True
                for j, (_, feature_list2) in enumerate(new_feature_bank.items()):
                    if i == j:
                        continue
                    for feature_set2 in feature_list2:
                        if new_feature_set.issubset(feature_set2):
                            to_add = False
                            break
                if to_add:
                    new_feature_bank[_class].append(new_feature_set)

    return new_feature_bank

def extand_feature_bank(feature_bank):
    # Apply extand1_feature_bank until no more changes are made
    new_feature_bank = extand1_feature_bank(feature_bank)
    while new_feature_bank != feature_bank:
        feature_bank = new_feature_bank
        new_feature_bank = extand1_feature_bank(feature_bank)
    return new_feature_bank

if __name__ == "__main__":
    feature_bank = {    
        "class_1": [{1, 2, 3}],
        "class_2": [{1, 3, 4}],
        "class_3": [{2, 3, 5}],
        "class_4": [{1, 2, 4, 5}],
        "class_5": [{2, 3, 4}]
    }

    feature_bank = extand_feature_bank(feature_bank)

    tree = Tree(build_tree(feature_bank))
    class_ = tree.forward([1, 2, 4, 5])
    tree.prnt()