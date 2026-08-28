##############################################################################################
##############################################################################################
##############################################################################################

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2

import numpy as np
import time
import craig


##############################################################################################
##############################################################################################
##############################################################################################


# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Linear(28*28, 100),
            nn.Sigmoid(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layer_stack(x)
        return logits


# Because keras uses glorot/xavier (uniform) initialisation with zero biases
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)


# Create an instance of the model, along with the loss function and optimiser
def init_model(learn_rate: float, weight_decay: float = 0):
    '''
    Initialise the device, model, loss function and optimiser.

    Parameters
    - learn_rate: float, the learning rate for the model
    - weight_decay: float, the L2 regularisation parameter for the model

    Returns
    - device: str, the device that PyTorch will use for computation
    - model: NeuralNetwork, the actual model to undergo training
    - loss_fn: a loss function
    - optimiser: an optimiser

    Notes
    - Currently, the loss function and optimiser are directly input here as CrossEntropyLoss and
      SGD; code could be expanded in future to support using other loss functions / optimisers
    '''

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    model.apply(init_weights)
    print(model)

    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, weight_decay=weight_decay)

    return device, model, loss_fn, optimizer


##############################################################################################
##############################################################################################
##############################################################################################


def load_datasets(dataset_name: str):
    '''
    Load in a particular dataset.

    Parameters
    - dataset_name: str, one of ['mnist'], the dataset to use

    Returns
    - training_data: training data for dataset `dataset_name`
    - test_data: test data for dataset `dataset_name`

    Notes
    - Supports expansion to support other datasets in the future
    '''
    training_data, test_data = None, None

    if dataset_name.lower() == 'mnist':
        # Download training data from open datasets.
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        )

        # Download test data from open datasets.
        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        )

    return training_data, test_data


##############################################################################################
##############################################################################################
##############################################################################################


# Define the training cycle
def train(device, model, loss_fn, optimizer, dataset, sample_weights, batch_size):
    '''
    The training phase of a particular epoch.

    Parameters
    - device: str, the device that PyTorch will use for computation
    - model: NeuralNetwork, the actual model to undergo training
    - loss_fn: a loss function
    - optimiser: an optimiser
    - dataset: the (subset) of data to be used for this epoch of training
    - sample_weights: torch.array, the per-sample weights to be used for this epoch of training
    - batch_size: int, the mini-batch size for the training
    '''

    # Perform a full training cycle
    dataloader = DataLoader(dataset, batch_size=batch_size)
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Obtain the appropriate set of sample weights for this batch
        index = batch * batch_size
        weights = sample_weights[index : index + len(X)]
        weights = weights.to(device)
        weights = weights.view(-1, 1, 1)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        loss = loss * weights
        loss = torch.mean((loss * weights))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Printing out some results in order to view progress
        if batch % 100 == 0:
            loss, current = loss.mean().item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


##############################################################################################
##############################################################################################
##############################################################################################


# The recomputation of weighted subsets of the data following a training cycle
def recompute_indices_and_weights(device, model, original_dataset, dataset, setting, 
                                  epoch, subset_size=1.0):
    '''
    Recompute the indicies and weights and construct the weighted subset for the next training phase.
    
    Parameters
    - device: str, the device that PyTorch will use for computation
    - model: NeuralNetwork, the actual model to undergo training
    - original_dataset: the original (training) dataset from which a new subset may be constructed
    - dataset: the subset used from the last completed training phase

    - setting: str | list[str],
        - if str, one of ['all', 'random', 'craig']
        - if list[str], must be a list as long as the number of epochs, all of whose elements
          are one of ['all', 'random', 'craig']
          
    - epoch: int, the epoch number (0-indexed)
    - subset_size: float, the required subset size

    Returns
    - subset: Subset, the required subset of the original dataset
    - (weights): torch.array, the per-sample weights associated with the dataset
    '''

    # Supports changing the setting with each epoch
    if isinstance(setting, list):
        setting = setting[epoch]

    # Use all the data
    if setting == 'all':
        return dataset, torch.ones(len(dataset.data))

    # Use random subsets of the data
    elif setting == 'random':
        indices = np.arange(0, len(original_dataset.data))
        np.random.shuffle(indices)
        indices = indices[:int(subset_size * len(original_dataset.data))]

        # Reload this new subset of the data to use on the next training cycle
        subset = Subset(original_dataset, indices)
        return subset, torch.ones(int(subset_size * len(original_dataset.data)))

    # Use the subsets of the data generated by the CRAIG algorithm
    elif setting == 'craig':
        all_training_data = DataLoader(original_dataset, batch_size=len(original_dataset.data))

        X_train, Y_train = None, None
        for X, y in all_training_data:
            X_train, Y_train = X.to(device), y.to(device)
            break

        # Pass all the data through the model once
        model.eval()
        with torch.no_grad():
            logits = model(X_train)
            probabilities = nn.functional.softmax(logits, dim = 1)

        X_train = torch.squeeze(X_train)
        X_train = np.asarray(X_train.reshape(60000, 784))
        Y_train = np.asarray(Y_train)

        num_classes = len(np.unique(y))
        features = probabilities - torch.nn.functional.one_hot(torch.from_numpy(Y_train), num_classes)

        # Obtain the facility location orders and generate the new subset of the data
        indices, sample_weights, _, _ = craig.get_orders_and_weights(
            features.numpy(), int(subset_size * len(original_dataset.data)), 'euclidean', Y_train)

        sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)

        # Reload this new subset of the data to use on the next training cycle
        subset = Subset(original_dataset, indices)
        return subset, torch.from_numpy(sample_weights)


##############################################################################################
##############################################################################################
##############################################################################################


# Define the test cycle
def test(device, model, loss_fn, dataloader, classes):
    '''
    The test phase of a particular epoch.

    Parameters
    - device: str, the device that PyTorch will use for computation
    - model: NeuralNetwork, the actual model to undergo training
    - loss_fn: a loss function
    - dataloader: Dataloader, the dataloader associated with the test data
    - classes: torch.array, the list of output classes

    Returns
    - test_loss: float, the test loss
    - accuracy: float, the overall test accuracy
    - class_accuracies: torch.array
    - FPR: torch.array, the class-wise false positive rates
    - FNR: torch.array, the class-wise false negative rates
    '''

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, accuracy = 0, 0
    class_counts = torch.zeros(len(classes))

    TP = torch.zeros(len(classes)) # true positive counts
    FP = torch.zeros(len(classes)) # false positive counts
    TN = torch.zeros(len(classes)) # true negative counts
    FN = torch.zeros(len(classes)) # false negative counts

    # Without tracking gradients, pass all test data through the model
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            # Compute the test loss
            test_loss += loss_fn(pred, y).mean().item()

            # Compute the accuracy (overall)
            correct = (pred.argmax(1) == y).type(torch.float)
            accuracy += correct.sum().item()

            # Compute the confusion matrix for each class `i`
            for i in classes:
                pred_i = (pred.argmax(1) == i).type(torch.float) # places where prediction is `i`
                y_i = (y == i).type(torch.float) # places where true vlaue is `i`

                ones = torch.ones_like(y) # for complements and shifts

                true_pos_i = pred_i * y_i
                false_pos_i = pred_i * (ones - y_i)
                true_neg_i = (ones - pred_i) * (ones - y_i)
                false_neg_i = (ones - pred_i) * y_i

                class_counts[i] += (true_pos_i + true_neg_i).sum().item()
                TP[i] += true_pos_i.sum().item()
                FP[i] += false_pos_i.sum().item()
                TN[i] += true_neg_i.sum().item()
                FN[i] += false_neg_i.sum().item()
    
    # Scale against the size of the test dataset
    test_loss /= num_batches
    accuracy /= size

    # Compute the accuracies, false positive and false negative rates per class
    class_accuracies = (TP + TN) / (TP + FP + TN + FN)
    FPR = FP / (FP + TN) # false positive rate
    FNR = FN / (FN + TP) # false negative rate

    print(f"Test Metrics: \n  Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")
    #print(f"  Class Accuracies: {class_accuracies.numpy()}")
    print(f"  FPRs: {FPR.numpy()}\n  FNRs: {FNR.numpy()}")

    # Return these values to store for later
    return test_loss, accuracy, class_accuracies, FPR, FNR


##############################################################################################
##############################################################################################
##############################################################################################

# Define the full sequence to train the model
def run(device, model, loss_fn, optimizer, training_data, test_data, setting, 
        epochs, batch_size, subset_size=1.0, save_model=False, to_collect=[]):
    '''
    A full model training run.

    Parameters
    - device: str, the device that PyTorch will use for computation
    - model: NeuralNetwork, the actual model to undergo training
    - loss_fn: a loss function
    - optimiser: an optimiser
    - training_data: the full training dataset
    - test_data: the full test dataset

    - setting: str | list[str],
        - if str, one of ['all', 'random', 'craig']
        - if list[str], must be a list as long as the number of epochs, all of whose elements
            are one of ['all', 'random', 'craig']

    - epochs: int, number of epochs of training
    - batch_size: int, the mini-batch size for the training
    - subset_size: float, the required subset size

    - save_model: bool, whether to save the trained model or not
    - to_collect: list[str], the metrics that should be recorded and stored
        - must be a subset of ['test_loss', 'accuracy', 'epoch_duration', 
                               'class_accuracies', 'fpr', 'fnr']

    Returns
    - data_collected: dict, a dictionary of all the collected data
    '''
    
    # Set up the training subset and sample weights
    subset_data = training_data
    sample_weights = torch.ones(len(subset_data.data))

    # Create the dataloader for the test data
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    all_test_data = DataLoader(test_data, batch_size=len(test_data.data))
    for X, y in all_test_data:
        classes = torch.unique(y)
        break

    # Set up what data should be collected
    all_possible = {
        'test_loss': 0,
        'accuracy': 0,
        'class_accuracies': len(classes),
        'fpr': len(classes),
        'fnr': len(classes),
        'epoch_duration': 0
    }

    data_collected = {}
    for element in to_collect:
        if all_possible[element] == 0: # storing scalar values
            data_collected[element] = np.zeros(epochs)
        else: # storing array values
            data_collected[element] = np.zeros((epochs, all_possible[element]))

    # Main run loop
    for t in range(epochs):
        start_time = time.time()
        print(f'-------------------------------\nEpoch {t+1}/{epochs}\n-------------------------------')

        train(device, model, loss_fn, optimizer, subset_data, sample_weights, batch_size)

        subset_data, sample_weights = recompute_indices_and_weights(device, model, training_data,
                                                                    subset_data, setting=setting,
                                                                    epoch=t, subset_size=subset_size)
        
        test_loss, accuracy, class_accuracies, fpr, fnr = test(device, model, loss_fn, test_dataloader, classes)

        epoch_duration = time.time() - start_time
        print(f'Epoch duration: {epoch_duration} \n')

        # Storing collected data
        for key, _ in all_possible.items():
            if key in to_collect:
                data_collected[key][t] = eval(key)

    if save_model:
        torch.save(model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")

    return data_collected


##############################################################################################
##############################################################################################
##############################################################################################


def save_data_to_file(filename, data):
    '''
    A helper function to save the necessary data to a file.

    Parameters:
    - filename: str, the filename where the data should be saved (will be a .npz file)
    - data: dict, the data to be saved
    '''

    command = f'np.savez("{filename}", '
    for key, value in data.items():
        command += f'{key}={value.tolist()}, '
    command = command[:-2] + ')'

    eval(command) # write the data


##############################################################################################
##############################################################################################
##############################################################################################

def collect_data(dataset, batch_size, subset_size, epochs,
                 learn_rate, weight_decay, setting, to_collect, runs, save_files):
    '''
    Conducting multiple independent runs to train a certain model and saving the data.

    Parameters
    - datase: str, one of ['mnist'], the dataset to use
    - batch_size: int, the mini-batch size for the training
    - subset_size: float, the required subset size
    - epochs: int, number of epochs of training
    - learn_rate: float, the learning rate for the model
    - weight_decay: float, the L2 regularisation parameter for the model

    - setting: str | list[str],
        - if str, one of ['all', 'random', 'craig']
        - if list[str], must be a list as long as the number of epochs, all of whose elements
            are one of ['all', 'random', 'craig']

    - to_collect: list[str], the metrics that should be recorded and stored
        - must be a subset of ['test_loss', 'accuracy', 'epoch_duration', 
                                'class_accuracies', 'fpr', 'fnr']

    - runs: int, number of independent repetitions of model training
    - save_files: list[str] with length `runs`, a list of filenames to which the collected
                  data should be saved
    '''

    # Load in the training and test data
    training_data, test_data = load_datasets(dataset)

    for i in range(runs):
        if isinstance(subset_size, list):
            subset_size = subset_size[i]

        # Initialise the model
        device, model, loss_fn, optimizer = init_model(learn_rate, weight_decay)

        # Train the model appropriately
        data_collected = run(device, model, loss_fn, optimizer, training_data, test_data,
            setting=setting, epochs=epochs, batch_size=batch_size, subset_size=subset_size,
            to_collect=to_collect)

        print(f'------------------------------- Completed -------------------------------')
        #print(data_collected)

        # Save the collected data to the given file (do not need to write .npz extension)
        save_file = save_files[i]
        save_data_to_file(save_file, data_collected)

