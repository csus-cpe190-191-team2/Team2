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
| <img src="images/Model_1_profile.jpg" alt="Model 1" width="50%" height="50%"> | <img src="images/Model_2_profile.jpg" alt="Model 1" width="50%" height="50%"> |

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




