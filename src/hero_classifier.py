from typing import List
import torch
import tqdm
import torch.optim as optim
import torch.nn as nn

from src.net import Net

class Classifier:
    def __init__(self, input_size: List[int], number_classes: int) -> None:
        self._net: Net = Net(input_size=input_size, number_classes=number_classes)

    def train(
            self,
            train_loader,
            validation_loader, 
            epochs: int, 
            criterion: nn.Module
    ) -> None:
        optimizer = optim.SGD(self._net.parameters(), lr=0.001, momentum=0.9)
        
        loader: tqdm = tqdm.tqdm(range(epochs))
        for epoch in loader:
            running_loss: float = 0.0
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self._net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
            
            correct: int = 0
            total: int = 0
            with torch.no_grad():
                for i, data in enumerate(validation_loader):
                    images, labels = data
                    # calculate outputs by running images through the network
                    outputs = self._net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
