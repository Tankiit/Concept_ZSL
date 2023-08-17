import torch
import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet = torchvision.models.resnet18().to(device)
resnet.fc = torch.nn.Linear(512, 10).to(device)

import torch.optim as optim

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=3e-4)

from torchmetrics import Accuracy
accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1).to(device)

EPOCHS = 15

from tqdm import tqdm

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    print(f"EPOCH: {epoch}")
    running_loss = 0.0
    running_acc = 0.0
    for i, data in tqdm(enumerate(trainloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:
            with torch.no_grad():
                for j, vdata in enumerate(testloader):
                    vinputs, vlabels = vdata
                    vinputs = vinputs.to(device)
                    vlabels = vlabels.to(device)
                    voutputs = resnet(vinputs)
                    running_acc += accuracy(voutputs, vlabels)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}, ACC: {running_acc / (j+1)}')
            running_loss = 0.0
            running_acc = 0.0

print('Finished Training')