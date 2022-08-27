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
# from torchvision import transforms
# from torchvision.utils import make_grid
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
root = "images/lanes/"


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
    # If the start button is pressed:
    # Stop the training session and turn off motors
    if curr_input == "START":
        if motor_control.motor_state is True:
            motor_control.toggle_motor()
        return False    # Ends the training loop

    # Drive motor using curr_input
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

    drive_label = motor_control.get_drive_state_label()

    # DEBUG OUTPUT
    if curr_input is not None:
        print(f'{curr_input}\t--->\t{drive_label}')

    return drive_label


# Creates motor state directories if missing
def data_dir_setup(label_dir, motor_handler: motor):
    drive_state_list = motor_handler.get_drive_state_label(return_all=True)

    for state in drive_state_list:
        path = os.path.join(label_dir, state)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


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
        sample_percent = int(input("\nEnter percent of samples to save for testing pool:\n>> "))
    else:
        sample_percent = 0

    # Initialize input device
    input_device = gamepad.Controller(3)

    # Initialize motor controller
    driver = motor.MotorControl()
    driver.set_speed(1)     # Set drive speed to MIN

    # Initialize train/test data path
    train_save_path = os.path.join(root, 'train')
    test_save_path = os.path.join(root, 'test')
    save_path = train_save_path     # Default save path
    data_dir_setup(train_save_path, driver)
    data_dir_setup(test_save_path, driver)

    # Initialize camera controller
    camera = eyes.Eyes()

    try:
        start_time = time.time()
        milli_sec_counter = time.time_ns()
        camera.camera_warm_up()

        while motor_state:
            curr_input = input_device.read_command()
            motor_state = controller_input_handler(curr_input, driver)

            if (motor_state != "stopped") \
                    and (motor_state != "backward") \
                    and (motor_state is not False):

                # If sampling enabled by user:
                # swap directory path between test and train
                # within desired percentage (sample_percent).
                if test_sample_collection:
                    if random.randrange(0, 100 + 1) < sample_percent:
                        save_path = test_save_path
                    else:
                        save_path = train_save_path

                # Set image name and path
                img_name = str(time.time_ns()) + ".png"
                img_save_path = os.path.join(save_path, motor_state, img_name)

                # Add delay between image captures to avoid data flooding
                # Save a forward image once every half second (1ms == 1^(6)ns)
                if motor_state == "forward":
                    if (time.time_ns() - milli_sec_counter) > 500000000:
                        camera.get_thresh_img()
                        camera.save_thresh_img(img_save_path)
                        drive_state_counter[motor_state] += 1
                        milli_sec_counter = time.time_ns()
                else:
                    # Save an image once every quarter second
                    if (time.time_ns() - milli_sec_counter) > 250000000:
                        camera.get_thresh_img()
                        camera.save_thresh_img(img_save_path)
                        drive_state_counter[motor_state] += 1
                        milli_sec_counter = time.time_ns()

    except KeyboardInterrupt:
        print("\nInterrupt detected. Operation stopped.\n")
    except TypeError as e:
        print("\nTypeError:", e)
    except AssertionError as e:
        print(e)

    # Run device destructors
    motor.destroy()
    input_device.kill_device()
    camera.destroy()

    # Print elapsed train time
    print(f'\nCompleted Data Collection.')
    print(f'\tElapsed Time: {round(((time.time()-start_time)/60), 2)} minutes')

    # Print tally of saved images from session
    print('\n-----------------')
    for drive_state_lbl, count in drive_state_counter.items():
        print(f'{drive_state_lbl}:', count)
    print('-----------------')


if __name__ == '__main__':
    while True:
        print("\nLane Detect CNN Menu [0-4]:\n")
        print("[0]: Collect Train Data")
        print("[4]: Exit")
        usr_input = int(input("\n>> "))

        # ### Collect Train Data
        if usr_input == 0:
            # Ask user for test data sampling
            enable_sampling = input("\nEnable train data sampling (y/n)?\n>> ")
            if enable_sampling.lower() == 'yes' or enable_sampling.lower() == 'y':
                enable_sampling = True
            else:
                enable_sampling = False

            # Start train data collection
            collect_train_data(enable_sampling)

        # ### Exit
        if usr_input == 4:
            break
