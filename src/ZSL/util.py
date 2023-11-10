def get_CUB_test_labels(path):
    with open(path, "r") as f:
        lines = f.readlines()
        
    return [int(line.split(".")[0]) for line in lines]

def get_AwA2_test_labels(path):
    with open(path, "r") as f:
        return [s.strip() for s in f.readlines()]

if __name__ == "__main__":
    print(get_CUB_test_labels("src/ZSL/splits/CUB/CUBtestclasses.txt"))
    print(get_AwA2_test_labels("src/ZSL/splits/AwA2testclasses.txt"))
    