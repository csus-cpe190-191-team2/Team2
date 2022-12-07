<!-- Sacramento State Logo and Title -->
<br />
<div align="center">
  <a href="https://www.csus.edu/">
    <img src="images/ECS_primary_hor_1p_3c_white_hdl_hires.jpg" alt="Logo" width="50%" height="50%">
  </a>

  <h1 align="center">Team 2</h1>

  <p align="center">
    CpE 190-191 - Senior Design:<br /><b><h3>Self-Driving Remote Controlled Car</h3></b>
    <br />
    <br />
    <a href="https://youtube.com/playlist?list=PLHFOvcVOvE2IlpS0oUwSvtJbdVLczmRdj">View Demo</a><br />
    <a href="https://youtube.com/playlist?list=PLHFOvcVOvE2IlpS0oUwSvtJbdVLczmRdj"><img src="https://img.youtube.com/vi/FD-_DXSRU5c/0.jpg" width="25%" height="25%"></a>
  </p>
</div>
<br />

<!-- ABOUT THE PROJECT -->
## About The Project
For the reason of safety, we utilized a scale model car design, no larger than a typical remote-controlled car to demonstrate our manual and full self-driving capabilities. Two different model car designs were used for testing and demonstration. The first design utilized a fused filament fabricator, structural parts could then be 3D printed and assembled. The second design was assembled using fabricated acrylic and prefabricated metal parts. The drive system on both cars uses a differential drive. The autonomous drive source code can be modified to support any vehicle of any size capable of wheeled locomotion. The core components of our self-driving car design consist of computer vision, object detection, distance sensing, a convolutional neural net and a single board computer capable of running Debian based Linux for CPU dependent tasks. 
<br />
| Model 1 | Model 2 |
| :----: |    :----:   |
| <img src="images/Model_1_profile.jpg" alt="Model 1" width="50%" height="50%"> | <img src="images/Model_2_profile.jpg" alt="Model 2" width="50%" height="50%"> |

### Built With
* <img src="https://img.shields.io/badge/Python-3.9.2-blue">
* <img src="https://img.shields.io/badge/OpenCV-4.5.5.62-brightgreen">
* <img src="https://img.shields.io/badge/PyTorch-1.12.1-red">
<br />

<!-- GETTING STARTED -->
## Getting Started
The following section will cover the minimum required hardware and software installation.

### Prerequisites
The minimum required hardware is:
* Single board computer (Raspberry Pi 4)
* Differential drive system
* USB Camera

Software:
* Pip3
  ```sh
  python -m pip3 install --upgrade pip
  ```

### Installation
1. Clone the repo
   ```sh
   git clone https://github.com/csus-cpe190-191-team2/Team2.git
   ```
2. Install required packages
   ```sh
   pip3 install -r requirements.txt
   ```
<br />

<!-- USAGE EXAMPLES -->
## Usage
### Prerequisites
Before running the main program, it is best to calibrate the camera which will remove and potential lense distortion from the image. This can be done by first printing and mounting the calibration checkerboard on a flat surface. The checkerboard can be retrieved from `/images/camera_calibration/test` there will be a few different formats, any one will work. The next step is running the `get_image.py` with the camera pointed towards the mounted checkerboard. The script will automatically capture 10 images with a five second delay between each capture. The best results can be achieved by slighty changing the camera angle for each image. Once that is complete and the captured images are deemed satisfactory, the `camera_calibration.py` script can be run to determine the camera's disotrtion matrix which is then saved and used to undistort the image through OpenCV. 
1. Capture camera calibration images
   ```sh
   python3 get_image.py
   ```
2. Determine camera distortion matrix
   ```sh
   python3 camera_calibration.py
   ```

### 1. Lane Detection Calibration
For best results, the lane detection threshold values should be calibrated every time the lighting conditions change. This can be achieved by running the `lane_detect_calibration.py` script and adjusting the HSV sliders until the two lane lines are appear as clear and defined white lines with little to no noise in the background. Once the lane lines are clearly detected, drag the "Save Settings" slider to the right and the hsv filter values will be saved to disk. The lane detect calibration menu can then be closed by pressing the "q" key. 
1. Start the lane detect calibration script
   ```sh
   python3 lane_detect_calibration.py
   ```
2. Move the HSV sliders until everything but the lane lines are filtered from the image
<img src="/images/hsv_sliders.png" alt="HSV Sliders" width="25%" height="25%">
3. Drag the "Save Settings" slider to the right, then press the "q" key to exit once the lanes are clearly visible
<img src="/images/lane_detect_calibration.png" alt="Lane Detect Calibration" width="50%" height="50%">

### 2. Build a Self-Driving Dataset
Before training the convolutional neural net, a driving dataset must first be created. Building a driving dataset can be achieved by first connecting a bluetooth controller to the single board computer. The gamepad mappings that are currently supported are a keyboard, PlayStation 5 controller and the 8BitDo. Once the gamepad is connected, the `lane_dataset_builder.py` script can then be run. The first prompt will ask the user if train data should be sampled, in most cases the answer should be yes (y). If yes was answered to the train sampling, the next prompt will ask what percentage of images should be saved to the test pool, usually 10-25% is a good range. Once all the prompts are answered, training will now begin. The script will pause from taking any pictures while the motors are stopped and will resume when the car is in motion. The motors can be toggled using the "A" key (see step 4 for generic controller mapping). As the car is in motion, the script will be taking photos from the camera and saving the image to the folder associated with the current drive state. Once training has concluded, the script can be stopped using the start/stop button on the gamepad.
1. Start the dataset builder script
   ```sh
   python3 lane_dataset_builder.py
   ```
2. Answer train sampling prompts
3. Toggle motors on by pressing the "A" button then drive around track (see step 4 for generic gamepad mapping)
4. End training by pressing the Start/Stop button on the gamepad

### 3. Train the Convolutional Neural Net (CNN)
Once a driving dataset is established, training of the convolutional neural net can begin. The training can be started by running the `lane_detect_CNN.py` script. Once started, a menu will be presented and one of four options can be selected. The first option is to train the model, this will begin training the neural net over the established dataset, both CPU and CUDA training is supported. Option two will allow a user to test the trained neural net against the test dataset at anytime, this will display the current accuracy of the trained model and will also display the first five incorrect results. Option three will allow the user to do a quick test using the trained model, this will take a single capture from the camera and ouput the inferred drive state. Option four will exit the program.
1. Run the CNN training script
   ```sh
   python3 lane_detect_CNN.py
   ```
2. Enter "1" after the prompt to being training the CNN using the established dataset
3. Enter "2" after the prompt to test the newly trained model against new data that was not trained on
4. Enter "3" after the prompt to test the newly trained model against the current camera input
5. Enter "4" after the prompt to exit the program

### 4. Run the Main Program
Now that the CNN is fully trained, the car will be capable of full self-driving. Before beginning the program a gamepad or keyboard must first be connected to the singleboard computer using bluetooth. The gamepad mappings that are currently supported are a keyboard, PlayStation 5 controller and the 8BitDo. Once the gamepad is connected, `main.py` can be executed to begin the main driving program. Once fully executed, the car will begin in a manual drive state where the user is in full control. A generic gamepad mapping of the drive commands can be seen below. At any moment the user may toggle between manual drive mode and full self-driving. While in self-driving mode the camera will feed its input to the trained CNN which will then control the current drive state roughly every millisecond. The user cannot control the motor state while in self-driving mode.
1. Start the main program
   ```sh
   python3 main.py
   ```
2. Manually control the car using the connected gamepad
<img src="/images/generic_gamepad_mapping.png" alt="Generic Gamepad Mapping" width="50%" height="50%">

3. Toggle from manual mode to full self-driving mode using the connected gamepad
4. Exit the program by pressing start/stop button on the connected gamepad
<br />

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
<br />

<!-- CONTACT -->
## Contact

* Ryan Aboueljoud - [LinkedIn](https://www.linkedin.com/in/ryan-aboueljoud) - [GitHub](https://github.com/RyanAboueljoud)
* Koby Barrett - [LinkedIn](https://www.linkedin.com/in/koby-barrett-74a22b1a6/) - [GitHub](https://github.com/BigolGuy)
* David Quintanilla - [LinkedIn](https://www.linkedin.com/in/david-quintanilla-233148a4/) - [GitHub](https://github.com/dquint54)
* Tuan Trinh - [LinkedIn](https://www.linkedin.com/in/tuan-trinh-957a5813a/) - [GitHub](https://github.com/tuantrinh152)

Project Link: [https://github.com/csus-cpe190-191-team2/Team2](https://github.com/csus-cpe190-191-team2/Team2)

Team 2 Group Photo:

<img src="/images/Team2_Group_photo.jpg" alt="Team 2 Group Photo" width="50%" height="50%">
<br />

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Professor Neal Levine](https://www.csus.edu/college/engineering-computer-science/electrical-engineering/meet-us/)
* [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
* [OpenCV Docs](https://docs.opencv.org/4.x/)
* [Python evdev](https://github.com/gvalkov/python-evdev)
* [PyTorch for Deep Learning](https://www.udemy.com/share/101rrK3@Q2TySuLbRoTaH8ukSCWHkAxNjDJwxNna_Ig8KVRtv01qHhT0G08vQhdYPtrb4OV9/)
* [Img Shields](https://shields.io)
* [Best README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right"><a href="#team-2">back to top</a></p>

