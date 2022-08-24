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
from torchvision import datasets, transforms, models # add models to the list
from torchvision.utils import make_grid
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import eyes
import controller as gamepad
import motor

import random

# Root path for train and test data
root = "/images/lanes/"


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


# Controls motors based on curr_input and returns drive state
def controller_input_handler(curr_input, motor_control: motor):
    # If the start button is pressed,
    # Stop the training session: turn off motors and clean GPIO assignments
    if curr_input == "START":
        motor_control.toggle_motor()
        return False    # Ends the training loop
    if curr_input == "A":
        motor_control.toggle_motor()
    if curr_input == "UP":
        motor_control.forward()
    if curr_input == "DOWN":
        motor_control.backward()
    if curr_input == "LEFT":
        motor_control.turn_left()
    if curr_input == "RIGHT":
        motor_control.turn_right()
    if curr_input == "LEFT TRIGGER":
        motor_control.rotate_left()
    if curr_input == "RIGHT TRIGGER":
        motor_control.rotate_right()

    return motor_control.get_drive_state_label()


# Creates a directory at label_dir path if it does not exist
def existing_dir(label_dir):
    if os.path.exists(label_dir):
        return True
    else:
        os.mkdir(label_dir)
        return False


# Collects CNN training data:
# User input is read in a loop which controls the motor.
# The current drive state will be saved to the corresponding folder
# which can then be used to train the CNNmodel.
def collect_train_data(test_sample_collection=False):
    # Initialize movement counter
    drive_state_counter = {
        "stopped": 0, "forward": 0, "backward": 0,
        "left": 0, "right": 0,
        "rotate_right": 0, "rotate_left": 0
    }

    motor_state = True

    # If test_sample_collection is enabled (TRUE),
    # ask for user input to determine what percent
    # of sampled data should be saved to the test data set
    if test_sample_collection:
        sample_percent = int(input("Enter percent of samples to save for testing pool:\n>> "))

    # Initialize train data path
    save_path = os.path.join(root, 'train')
    existing_dir(save_path)

    # Initialize input device
    input_device = gamepad.Controller
    input_device.set_controller(3)
    input_device.set_map()

    # Initialize motor controller
    driver = motor.MotorControl
    driver.set_speed(1)     # Set drive speed to MIN

    # Initialize camera controller
    camera = eyes.Eyes
    camera.camera_warp_up()

    start_time = time.time()

    while motor_state:
        curr_input = input_device.read_command()
        motor_state = controller_input_handler(curr_input, driver)

        if motor_state != "stopped":
            # If enabled, swap directory path between test and train
            # based on desired percentage (sample_percent).
            if test_sample_collection:
                if random.randrange(0, 100 + 1) < sample_percent:
                    save_path = os.path.join(root, 'test')
                else:
                    save_path = os.path.join(root, 'train')

            # Update threshold camera frame and name
            camera.get_thresh_img()
            img_name = str(time.time_ns()) + ".png"

            # Save image to path associated with drive state
            img_save_path = os.path.join(save_path, motor_state)
            existing_dir(img_save_path)
            img_save_path = os.path.join(img_save_path, img_name)
            camera.save_thresh_img(img_save_path)
            drive_state_counter[motor_state] += 1

    # Run device destructors
    motor.destroy()
    input_device.kill_device()
    camera.destroy()

    # Print elapsed train time
    print(f'Completed Data Collection.')
    print(f'\tElapsed Time: {time.time()-start_time/60} minutes')

    # Print tally of saved images from session
    print("\nNew Images:\n")
    for drive_state_lbl, count in drive_state_counter.items():
        print(drive_state_lbl, count)


if __name__ == '__main__':
    while True:
        print("Choose:")
        print("[0]: Collect Train Data")
        print("[4]: Exit")
        curr_input = int(input(">> "))

        # Collect Train Data
        if curr_input == 0:
            enable_sampling = input("Enable train data sampling(yes/no)?\n>> ")
            if enable_sampling.lower() == 'yes':
                enable_sampling = True
            else:
                enable_sampling = False

            collect_train_data(enable_sampling)

        # Exit
        if curr_input == 4:
            break
