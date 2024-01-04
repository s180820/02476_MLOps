"""When this file runs, it should take the raw data e.g. the corrupted MNIST files from yesterday which now should 
be located in a data/raw folder and process them into a single tensor, normalize the tensor and save this intermediate 
representation to the data/processed folder. By normalization here we refer to making sure the images have mean 0 
and standard deviation 1."""

import torch
from torchvision import datasets, transforms

if __name__ == "__main__":
    # Get the data and process it
    mnist_train = datasets.MNIST("data/raw", train=True, download=True)
    mnist_test = datasets.MNIST("data/raw", train=False, download=True)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])

    # transfrom the data
    mnist_train.transform = transform
    mnist_test.transform = transform

    # save the data
    torch.save(mnist_train, "data/processed/training.pt")
    torch.save(mnist_test, "data/processed/test.pt")
