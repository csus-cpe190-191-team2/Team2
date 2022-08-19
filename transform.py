import os
import cv2
import numpy as np
from tqdm import tqdm

IMG_HEIGHT = 240
IMG_WIDTH = 480
LANE_ROOT = '../images/lane'
LANE_TRAIN_PATH = '../images/lanes/train'  ### outside of folder scope
LANE_TEST_PATH = '../images/lanes/test'

class DataControl:
    stopped = LANE_TRAIN_PATH+'/stopped'
    forward = LANE_TRAIN_PATH+'/forward'
    backward = LANE_TRAIN_PATH+'/backward'
    left = LANE_TRAIN_PATH+'/left'
    right = LANE_TRAIN_PATH+'/right'
    Rright = LANE_TRAIN_PATH+'/Rright'
    Rleft = LANE_TRAIN_PATH+'/Rleft'

    LABELS = {
        stopped: 0
        , forward: 1
        , backward: 2
        , left: 3
        , right: 4
        , Rright: 5
        , Rleft: 6
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
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        ###or return self.training_data

    def collect_data(self, set='train', ):
        save_path = os.path.join(LANE_ROOT, set)





