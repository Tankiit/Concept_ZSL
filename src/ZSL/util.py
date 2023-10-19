def get_CUB_test_labels(path):
    with open(path, "r") as f:
        lines = f.readlines()
        
    return [int(line.split(".")[0]) for line in lines]

if __name__ == "__main__":
    print(get_CUB_test_labels("src/ZSL/CUBtestclasses.txt"))