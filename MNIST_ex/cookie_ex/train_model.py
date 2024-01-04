"""Used to train the model"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model import MyNeuralNet

import matplotlib.pyplot as plt

epochs = 5

TRAIN_PATH = "data/processed/test.pt"
TEST_PATH = "data/processed/training.pt"

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler

train_data = torch.load(TRAIN_PATH)
test_data = torch.load(TEST_PATH)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)


def validation(model, testloader, criterion):
    """Validate the model on the testdata by calculating the sum of mean loss and mean accuracy for each test batch.

    Arguments:
        model: torch network
        testloader: torch.utils.data.DataLoader, dataloader of test set
        criterion: loss function
    """
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def train(model, trainloader, testloader, criterion, optimizer=None, epochs=5, print_every=40):
    """Train model."""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)

                print(
                    "Epoch: {}/{}.. ".format(e + 1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                    "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
                )

                # plot the loss

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()


if __name__ == "__main__":
    model = MyNeuralNet(784, 10, [512, 256, 128])
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    train(model, trainloader, testloader, criterion, optimizer, epochs)
    torch.save(model.state_dict(), "models/model.pth")
