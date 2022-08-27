import os
import cv2
import numpy as np
from tqdm import tqdm
import motor as m

IMG_HEIGHT = 240
IMG_WIDTH = 480
LANE_ROOT = '../images/lane'
LANE_TRAIN_PATH = '../images/lanes/train'  ### outside of folder scope
LANE_TEST_PATH = '../images/lanes/test'

class DataControl:
    stopped = 'stopped'
    forward = 'forward'
    backward = 'backward'
    left = 'left'
    right = 'right'
    Rright = 'Rright'
    Rleft = 'Rleft'

    LABELS = {
        stopped: 0
        , forward: 1
        , backward: 2
        , left: 3
        , right: 4
        , Rright: 5
        , Rleft: 6
    }

    INV_LABELS = {
        0: stopped
        , 1: forward
        , 2: backward
        , 3: left
        , 4: right
        , 5: Rright
        , 6: Rleft
    }

    training_data = []

    stopcnt = 0
    fwdcnt = 0
    backcnt = 0
    leftcnt = 0
    rightcnt = 0
    Rrightcnt = 0
    Rleftcnt = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            label = os.path.join(LANE_TRAIN_PATH, label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = img.resize(img, (IMG_WIDTH,IMG_HEIGHT)) ### these might be switched
                    ### use one hot vector
                    self.training_data.append([np.array(img), np.eye(7)[self.LABELS[label]]])

                    if label == self.stopped:
                        self.stopcnt += 1
                    elif label == self.forward:
                        self.fwdcnt += 1
                    elif label == self.backward:
                        self.backcnt += 1
                    elif label == self.left:
                        self.leftcnt += 1
                    elif label == self.right:
                        self.rightcnt += 1
                    elif label == self.Rright:
                        self.Rrightcnt += 1
                    elif label == self.Rleft:
                        self.Rleftcnt += 1
                except Exception as e:
                    print('error')
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        ###or return self.training_data

    def collect_data(self, img, drive_state, data_set='train'):
        save_path = os.path.join(LANE_ROOT, data_set)
        self.existing_dir(save_path)
        save_path = os.path.join(save_path, self.INV_LABELS[drive_state])
        if not self.existing_dir(save_path):
            file_name = os.path.join(save_path, '1.png')
            cv2.imwrite(file_name, img)
        else:
            last_img_num = int(os.listdir(save_path)[-1].strip('.png'))
            last_img_num += 1
            last_img_num = str(last_img_num)
            file_name = last_img_num + '.png'
            file_name = os.path.join(save_path, file_name)
            cv2.imwrite(file_name, img)

    def existing_dir(self, label_dir):
        if os.path.exists(label_dir):
            if len(os.listdir(label_dir)) == 0:
                return False #nothing in dir
            else:
                return True
        else:
            os.mkdir(label_dir)
            return False #nothing in dir


