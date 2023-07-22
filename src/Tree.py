from math import factorial, ceil, log2

import torch
import torch.nn as nn
def get_dataset(feature_vector, feature_bank):
	inputs = []
	labels = []

	# inputs is a list of sums of feature vectors
	for _class, feature_list in feature_bank.items():
		for feature_set in feature_list:
			inputs.append(feature_vector[list(feature_set)].sum(dim=0))
			labels.append(int(_class.split("_")[1])-1)

	return torch.stack(inputs), torch.tensor(labels)

def build_network(feature_vector, feature_bank):
		"""
		Calculate required network size, and build the network
		Train the network until 100% accuracy
		Make custom loss function to maximize value on class index and set all other values to 0 (can also try with using cross entropy loss)
		"""
		features, dim = feature_vector.shape
		classes = len(feature_bank)

		max_features = 0
		for feature_list in feature_bank.values():
			for feature_set in feature_list:
				max_features = max(max_features, len(feature_set))

		combinations = factorial(features) / (factorial(max_features) * factorial(features - max_features))

		hidden = ceil(log2(combinations))
		# find closest multiple of 64
		hidden = int((hidden + 63) // 64 * 64)

		# build the network
		network = nn.Sequential(
			nn.Linear(dim, hidden),
			nn.ReLU(),
			nn.Linear(hidden, classes),
		)

		# get dataset
		inputs, labels = get_dataset(feature_vector, feature_bank)

		# train the network
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

		while True:
			optimizer.zero_grad()
			outputs = network(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			predicted = torch.argmax(outputs, dim=1)
			correct = (predicted == labels).sum().item()
			if correct == len(labels):
				break

		return network, max_features

class Tree:
	def __init__(self, root):
		self.root = root

	def forward(self, x):
		# x is a list of feature ids
		return self.root.forward(x)
	
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

def get_direct_subsets(feature_set):
	# feature_set is a set of features
	# return a list of all subsets of length smaller by 1
	subsets = []

	for feature in feature_set:
		new_feature_set = feature_set.copy()
		new_feature_set.remove(feature)
		subsets.append(new_feature_set)

	return subsets

from tqdm import tqdm
def extand_feature_bank(feature_bank):
	"""
	Go class by class, generate all subsets of length smaller by 1, if not a subset of any other class, add to the class.
	If no such subset exists, stop.
	"""

	for class_, feature_list in tqdm(feature_bank.items()):
		new_feature_list = [feature_list[0]]
		source_feature_sets = [feature_list2[0] for class2, feature_list2 in feature_bank.items() if class_ != class2]

		subsets = get_direct_subsets(feature_list[0])
		while len(subsets) > 0:
			new_subsets = []
			for subset in subsets:
				to_add = True
				for source_feature_set in source_feature_sets:
					if subset.issubset(source_feature_set):
						to_add = False
						break
				if to_add and subset not in new_feature_list:
					new_feature_list.append(subset)
					new_subsets += get_direct_subsets(subset)
			subsets = new_subsets

		feature_bank[class_] = new_feature_list

	return feature_bank

if __name__ == "__main__":
	feature_bank = {    
		"class_1": [{1, 2, 3}],
		"class_2": [{1, 3, 4}],
		"class_3": [{2, 3, 5}],
		"class_4": [{1, 2, 4, 5}],
		"class_5": [{2, 3, 4}]
	}

	feature_bank = extand_feature_bank(feature_bank)
	print(feature_bank)

	tree = Tree(build_tree(feature_bank))

	DIM = 32
	FEATURES = 6
	feature_vector = torch.randn((FEATURES, DIM))