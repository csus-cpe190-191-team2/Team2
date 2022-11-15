import os
import time
import eyes
import random

# Root path for train and test data
root = "images/objects/"

# Creates motor state directories if missing
def data_dir_check(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# Collects CNN training data:
# User input is read in a loop which controls the motor.
# The current drive state will be saved to the corresponding folder
# which can then be used to train the CNNmodel.
def collect_train_data(test_sample_collection=False):
    train_save_path = os.path.join(root, 'train')
    test_save_path = os.path.join(root, 'test')
    save_path = train_save_path  # Default save path
    data_dir_check(train_save_path)
    data_dir_check(train_save_path)
    motor_state = True

    # Initialize movement counter
    sign_counter = [
        0, 0, 0
    ]

    if test_sample_collection:
        sample_percent = int(input("\nEnter percent of samples to save for testing pool:\n>> "))
    else:
        sample_percent = 0

    # Initialize train/test data path
    train_save_path = os.path.join(root, 'train')
    test_save_path = os.path.join(root, 'test')
    save_path = train_save_path     # Default save path
    data_dir_check(train_save_path)
    data_dir_check(train_save_path)

    # Initialize camera controller
    camera = eyes.Eyes()

    start_time = time.time()

    try:
        camera.camera_warm_up()

        while True:
            get_image = input("Get Image? (y/n)\n>> ")
            if get_image.lower() == 'n':
                break
            while True:
                camera.cap_img()
                # eye.show_img(camera.img, "Camera View", 1)
                eyes.cv2.imshow("Camera View", camera.img)
                if eyes.cv2.waitKey(0):
                    # IF 1 OR 2 OR 3 SAVE IMAGE TO APPROPRIATE FOLDER
                    get_image = input("Keep Image? (y/n)\n>> ")
                    if get_image.lower() == "y":
                        # If sampling enabled by user:
                        # swap directory path between test and train
                        # within desired percentage (sample_percent).
                        if random.randrange(0, 100 + 1) < sample_percent:
                            save_path = test_save_path
                        else:
                            save_path = train_save_path
                        get_object = input("Classify Object (1-3):\n1: Stop Sign\n2: Speed Sign\n3:Yield Sign\n>> ")
                        if get_object == "1":
                            save_path = os.path.join(save_path, "stop")
                        elif get_object == "2":
                            save_path = os.path.join(save_path, "speed")
                        elif get_object == "3":
                            save_path = os.path.join(save_path, "yield")
                        else:
                            continue
                        data_dir_check(save_path)
                        # Set image name and path
                        img_name = str(time.time_ns()) + ".png"
                        save_path = os.path.join(save_path, img_name)
                        camera.get_grayscale_img()
                        camera.save_bw_img(save_path)

                        sign_counter[int(get_object) - 1] += 1

                    else:
                        time.sleep(5)
                        continue

                    eyes.cv2.destroyAllWindows()
                    break

    except KeyboardInterrupt:
        print("\nInterrupt detected. Operation stopped.\n")
    except TypeError as e:
        print("\nTypeError:", e)
    except AssertionError as e:
        print(e)
    print(f'current counter total:\nstop: {sign_counter[0]}\nspeed: {sign_counter[1]}\nyield: {sign_counter[2]}')
    # Run device destructors
    camera.destroy()

if __name__ == '__main__':

    collect_train_data()

