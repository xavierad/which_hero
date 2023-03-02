import torchvision
import torch
import torch.nn as nn
from torchvision import transforms

from src.hero_classifier import Classifier

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()

hero_classifier: Classifier = Classifier(input_size=[6, 5], number_classes=10)
hero_classifier.train(train_loader=trainloader, validation_loader=testloader, epochs=10, criterion=criterion)