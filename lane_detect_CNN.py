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
# from torchvision.utils import make_grid
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
        # 480=(((480-2+1)/2)-2+1)/2 = 119.25
        # 240=(((240-2+1)/2)-2+1)/2 = 59.25
        self.fc1 = nn.Linear(119 * 59 * 16, 112)
        self.fc2 = nn.Linear(112, 6)

    def forward(self, X):
        X = F.max_pool2d(F.relu(self.conv1(X)), 2, 2)
        X = F.max_pool2d(F.relu(self.conv2(X)), 2, 2)
        X = X.view(-1, 119 * 59 * 16)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)


# Train the CNNmodel using saved image data
# collected from collect_train_data()
def train(train_existing=False):
    epochs = 3  # Number of training epochs
    epoch_count = 0     # epoch counter initializer
    epoch_start = 0     # Start number for epoch counter
    train_losses = []
    train_correct = []

    # Determine size of training dataset
    training_dataset_size = 0
    lst = os.listdir(os.path.join(root, 'train'))
    for i in lst:
        folder_contents = os.listdir(os.path.join(root, 'train', i))
        training_dataset_size += len(folder_contents)

    # Create an instance of the CNN model
    model = CNNmodel(use_gpu=torch.cuda.is_available())

    # Enable CUDA if available
    if model.use_gpu:
        print("Using CUDA")
        model = model.cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Load saved model data if file exists
    # AND either train_existing is true
    # OR previous training did not complete
    if os.path.exists(saved_model_fn):
        checkpoint = torch.load(saved_model_fn)
        epoch = checkpoint['epoch']
        # if epoch < epochs:
        #     train_correct = checkpoint['correct']
        #     epoch_start = epoch

        if (epoch < epochs) or train_existing:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss = checkpoint['loss']

            if epoch != epochs:
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

    print("Training started...")
    start_time = time.time()

    try:
        for i in range(epoch_start, epochs):
            trn_corr = 0    # Reset train correct count
            epoch_count = i
            for b, (X_train, y_train) in enumerate(train_loader):
                b += 1

                # Limit batches if desired
                # if b == 2:
                #     break

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
                            batch: {b:4} [{10*b:6}/{training_dataset_size}]  \
                            loss: {loss.item():10.8f}  \
                            accuracy: {trn_corr.item()*100/(10*b):7.3f}%')

            train_losses.append(loss)
            train_correct.append(trn_corr)
            print(f'{i + 1} of {epochs} completed')
            epoch_count = i+1
    except KeyboardInterrupt as interrupt:
        print(interrupt)

    print(epoch_count)

    run_time = (time.time() - start_time)/60
    print(f'\nDuration: {run_time:.2f} minute(s)')  # print the time elapsed

    # Save trained CNN model plus losses and correct count to disk
    # np.savez(train_losses_fn, train_losses)
    # np.savez(train_correct_fn, train_correct)
    # torch.save(model.state_dict(), saved_model_fn)
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch_count,
        'loss': train_losses,
        'correct': train_correct
    }
    torch.save(state, saved_model_fn)
    print(f'Model saved to \'{saved_model_fn}\'')



if __name__ == '__main__':
    while True:
        print("\nLane Detect CNN Menu [0-4]:\n")
        print("[1]: Train the model")
        print("[2]: Test the model")
        print("[3]: Evaluate Model Performance")
        print("[4]: Exit")
        try:
            usr_input = int(input("\n>> "))
        except UnicodeDecodeError as e:
            print('\n', e)
            usr_input = int(input("\n>> "))

        # ### Train CNN model
        if usr_input == 1:
            print("\nTrain on saved model? (y/n)")
            usr_input = input("\n>> ")
            if usr_input == 'yes' or usr_input == 'y':
                usr_input = True
            else:
                usr_input = False
            train(train_existing=usr_input)

        # ### Exit
        else:
            break
