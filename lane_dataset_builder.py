import os
import time

import eyes
import controller as gamepad
import motor

import random


# Root path for train and test data
root = "images/lanes/"


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
def data_dir_check(path):
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
    data_dir_check(train_save_path)
    data_dir_check(train_save_path)

    # Initialize camera controller
    camera = eyes.Eyes()

    start_time = time.time()
    milli_sec_counter = time.time_ns()
    try:
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
                img_path = os.path.join(save_path, motor_state)
                img_save_path = os.path.join(img_path, img_name)

                # Add delay between image captures to avoid data flooding
                # Save a forward image once every half second (1ms == 1^(6)ns)
                if motor_state == "forward":
                    if (time.time_ns() - milli_sec_counter) > 500000000:
                        camera.get_thresh_img()
                        data_dir_check(img_path)
                        camera.save_thresh_img(img_save_path)
                        drive_state_counter[motor_state] += 1
                        milli_sec_counter = time.time_ns()
                else:
                    # Save an image once every quarter second
                    if (time.time_ns() - milli_sec_counter) > 250000000:
                        camera.get_thresh_img()
                        data_dir_check(img_path)
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
    # ### Collect Train Data
    # Ask user for test data sampling
    enable_sampling = input("\nEnable train data sampling (y/n)?\n>> ")
    if enable_sampling.lower() == 'yes' or enable_sampling.lower() == 'y':
        enable_sampling = True
    else:
        enable_sampling = False

    # Start train data collection
    collect_train_data(enable_sampling)
