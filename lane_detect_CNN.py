"""
#   Convolutional Neural Network (CNN)
#   meant for predicting and training directional output
#   with the goal of driving within a lane.
#   Class supports training, testing and predicting.
#
#   Input must be a camera image that has
#   been filtered to a binary image,
#   highlighting the lane features.
#
#   Output is an array containing a
#   directional integer index value
#   and associated string key value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


root = 'images/lanes/'  # path for train and test data
train_losses_fn = 'vars/train_losses.npz'
train_correct_fn = 'vars/train_correct.npz'
test_losses_fn = 'vars/test_losses.npz'
test_correct_fn = 'vars/test_correct.npz'
saved_model_fn = 'vars/Lane_Detect-CNN-Model.pt'


# Convolutional Neural Network (CNN) class model
# Expects to work on binary image 240x480
class CNNmodel(nn.Module):
    def __init__(self, use_gpu=False):
        super().__init__()
        self.use_gpu = use_gpu
        self.conv1 = nn.Conv2d(1, 6, 2, 1)
        self.conv2 = nn.Conv2d(6, 16, 2, 1)
        # After two max pooling kernel=2, stride=2:
        # 480=(((480-2+1)/2)-2+1)/2 = 119.25, 240=(((240-2+1)/2)-2+1)/2 = 59.25
        self.fc1 = nn.Linear(119 * 59 * 16, 112)
        self.fc2 = nn.Linear(112, 3)    # left, right, forward

    def forward(self, X):
        X = F.max_pool2d(F.relu(self.conv1(X)), 2, 2)
        X = F.max_pool2d(F.relu(self.conv2(X)), 2, 2)
        X = X.view(-1, 119 * 59 * 16)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)


# Train the CNNmodel using saved image data
# collected from lane_dataset_builder.py
def train(train_existing=False, graph_data=False):
    epochs = 3  # Number of training epochs
    epoch_count = 0     # epoch counter initializer
    epoch_start = 0     # Start number for epoch counter
    train_correct = []
    train_losses = []
    test_correct = []
    test_losses = []

    # Create an instance of the CNN model
    model = CNNmodel(use_gpu=torch.cuda.is_available())

    # Enable CUDA if available
    if model.use_gpu:
        print("Using CUDA")
        model = model.cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load saved model data if file exists.
    # Also checks for train model resuming and loading
    if os.path.exists(saved_model_fn):
        checkpoint = torch.load(saved_model_fn)
        epoch = checkpoint['epoch']
        if (epoch < epochs) or train_existing:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss = checkpoint['loss']
            # If earlier training was cancelled: resume
            if epoch < epochs:
                train_correct = checkpoint['correct']
                epoch_start = epoch

            model.eval()
            print(f'Saved model loaded from \'{saved_model_fn}\'')
    else:
        print(f'No saved model found at \'{saved_model_fn}\'')

    # Build data loaders
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    train_data = \
        datasets.ImageFolder(os.path.join(root, 'train'), transform=transform)
    train_loader = \
        DataLoader(train_data, batch_size=10, shuffle=True, pin_memory=model.use_gpu)
    if graph_data:
        test_data = \
            datasets.ImageFolder(os.path.join(root, 'test'), transform=transform)
        test_loader = \
            DataLoader(test_data, batch_size=10, shuffle=True, pin_memory=model.use_gpu)

    print("Training started...")
    start_time = time.time()
    try:
        for i in range(epoch_start, epochs):
            trn_corr = 0    # Reset train correct count
            tst_corr = 0    # Reset test correct count
            epoch_count = i
            for b, (X_train, y_train) in enumerate(train_loader):
                b += 1

                # Pin tensor to GPU if CUDA is available
                if model.use_gpu:
                    X_train = X_train.cuda()
                    y_train = y_train.cuda()

                # Apply the model
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)

                # Tally the number of correct predictions
                predicted = torch.max(y_pred.data, 1)[1]
                batch_corr = (predicted == y_train).sum()
                trn_corr += batch_corr

                # Update the Parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print interim results
                if b%10 == 0:
                    print(f'epoch: {i+1:2}/{epochs}  \
                            batch: {b:4} [{10*b:6}/{len(train_data)}]  \
                            loss: {loss.item():10.8f}  \
                            accuracy: {trn_corr.item()*100/(10*b):7.3f}%')

            train_losses.append(loss)
            train_correct.append(trn_corr)

            if graph_data:
                # Run the testing batches
                with torch.no_grad():
                    for X_test, y_test in test_loader:
                        # Pin tensor to GPU if CUDA is available
                        if model.use_gpu:
                            X_test = X_test.cuda()
                            y_test = y_test.cuda()

                        # Apply the model
                        y_val = model(X_test)

                        # Tally the number of correct predictions
                        predicted = torch.max(y_val.data, 1)[1]
                        tst_corr += (predicted == y_test).sum()
                loss = criterion(y_val, y_test)
                test_losses.append(loss)
                test_correct.append(tst_corr)

            print(f'{i + 1} of {epochs} completed')
            epoch_count = i+1

    except KeyboardInterrupt as interrupt:
        print(interrupt)

    print(f'Test accuracy: {test_correct[-1].item() * 100 / len(test_data):.3f}%')

    # print the time elapsed
    run_time = (time.time() - start_time)/60
    print(f'\nDuration: {run_time:.2f} minute(s)')

    # Save trained CNN model plus losses and correct count to disk
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch_count,
        'loss': train_losses,
        'correct': train_correct
    }
    torch.save(state, saved_model_fn)
    print(f'Model saved to \'{saved_model_fn}\'')

    if graph_data:
        if model.use_gpu:
            for i, losses in enumerate(train_losses):
                train_losses[i] = losses.cpu()
            for i, losses in enumerate(test_losses):
                test_losses[i] = losses.cpu()
            for i, correct in enumerate(train_correct):
                train_correct[i] = correct.cpu()
            for i, correct in enumerate(test_correct):
                test_correct[i] = correct.cpu()

        train_losses = [loss.detach().numpy() for loss in train_losses]
        fig, axs = plt.subplots(2)
        fig.suptitle('Stacked loss and accuracy comparisons')
        axs[0].plot(train_losses, label='training loss')
        axs[0].plot(test_losses, label='validation loss')
        axs[0].set_title('Loss at the end of each epoch')
        axs[0].set(xlabel='Epochs', ylabel='Loss')
        axs[0].set_ylim(ymin=0)
        axs[0].set_xlim(xmin=0)
        axs[0].legend()
        axs[0].autoscale()

        axs[1].plot([t / 500 for t in train_correct], label='training accuracy')
        axs[1].plot([t / 100 for t in test_correct], label='validation accuracy')
        axs[1].set_title('Accuracy at the end of each epoch')
        axs[1].set(xlabel='Epochs', ylabel='% Accuracy')
        axs[1].set_ylim(ymin=0)
        axs[1].set_xlim(xmin=0)
        axs[1].legend()
        axs[1].autoscale()

        plt.subplots_adjust(hspace=0.8)
        plt.show()


def test(view_misses=False):
    # Create an instance of the CNN model
    model = CNNmodel(use_gpu=torch.cuda.is_available())

    # Enable CUDA if available
    if model.use_gpu:
        print("Using CUDA")
        model = model.cuda()

    # If no trained model exists: train a new one
    if not os.path.exists(saved_model_fn):
        print(f'No saved model found at \'{saved_model_fn}\'')
        print('Creating new trained model now.')
        train()

    # Load saved model from disk
    checkpoint = torch.load(saved_model_fn)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'Saved model loaded from \'{saved_model_fn}\'')

    # Build data loaders
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    test_data = \
        datasets.ImageFolder(os.path.join(root, 'test'), transform=transform)
    test_loader = \
        DataLoader(test_data, batch_size=10000, shuffle=True, pin_memory=model.use_gpu)

    # Run the testing batches
    print("Starting test batches...")
    with torch.no_grad():
        tst_corr = 0
        for X_test, y_test in test_loader:
            # Pin tensor to GPU if CUDA is available
            if model.use_gpu:
                X_test = X_test.cuda()
                y_test = y_test.cuda()

            # Apply the model
            y_val = model(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    print('Complete.')
    print(f'\nTest accuracy: {tst_corr.item()}/{len(test_data)} = {tst_corr.item() * 100 / (len(test_data)):7.3f}%')

    if view_misses:
        misses = np.array([])
        for i in range(len(predicted.view(-1))):
            if predicted[i] != y_test[i]:
                misses = np.append(misses, i).astype('int64')

        rows = 5
        row = iter(np.array_split(misses, len(misses)//rows+1))

        # widen the printed array
        np.set_printoptions(formatter=dict(int=lambda x: f'{x:5}'))

        if model.use_gpu:
            y_test = y_test.cpu()
            predicted = predicted.cpu()
            X_test = X_test.cpu()

        nextrow = next(row)
        lbls = y_test.index_select(0, torch.tensor(nextrow)).numpy()
        gues = predicted.index_select(0, torch.tensor(nextrow)).numpy()
        print(f"\nMissed Prediction Index (first {rows} rows):\n", nextrow)
        print("\nData Label Index:")
        print("Label: ", *np.array([get_label(i) for i in lbls]))
        print("\nPredictions:")
        print("Guess: ", *np.array([get_label(i) for i in gues]))

        images = X_test.index_select(0, torch.tensor(nextrow))
        plt.figure(figsize=(8, 6))
        im = make_grid(images, nrow=rows)
        plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
        plt.title(label=f'First {rows} misses')
        plt.show()


def get_label(item_index, dataset_type='train'):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    data = \
        datasets.ImageFolder(os.path.join(root, dataset_type), transform=transform)
    class_names = list(enumerate(data.classes))
    return class_names[item_index]


if __name__ == '__main__':
    while True:
        print("\nLane Detect CNN Menu [0-3]:\n")
        print("[1]: Train the model")
        print("[2]: Test the model")
        print("[3]: Exit")

        try:
            usr_input = int(input("\n>> "))
        except UnicodeDecodeError as e:
            print('\n', e)
            usr_input = 0
        except ValueError as e:
            usr_input = 0

        if usr_input == 1:      # ### Train CNN model
            train(train_existing=False, graph_data=True)
        elif usr_input == 2:    # ### Train CNN model
            test(view_misses=True)
        else:                   # ### Exit
            break
