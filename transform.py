import os
import cv2
import numpy as np
from tqdm import tqdm
#import motor as m
import matplotlib.pyplot as plt
from matplotlib import style

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

    # IMG_SIZE = 50
    CATS = "../catsvdogs/Cat" #"../catsvdogs/PetImages/Cat"
    DOGS = "../catsvdogs/Dog" #"../catsvdogs/PetImages/Dog"
    TESTING = "../catsvdogs/Testing" #"../catsvdogs/PetImages/Testing"
    CD_LABELS = {CATS: 0, DOGS: 1}
    INV_CD_LABELS = {0: 'cat', 1: 'dog'}

    catcount = 0
    dogcount = 0

    def make_training_data(self):
        # for label in self.LABELS:
        for label in self.CD_LABELS:
            print(label)
            #label = os.path.join(LANE_TRAIN_PATH, label)
            for f in tqdm(os.listdir(label)):
                #if "png" in f:
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) ### these might be switched
                        ### use one hot vector
                        # self.training_data.append([np.array(img), np.eye(7)[self.LABELS[label]]])
                        self.training_data.append([np.array(img), np.eye(2)[self.CD_LABELS[label]]])

                        if label == self.CATS:
                            self.catcount += 1
                        elif label == self.DOGS:
                            self.dogcount += 1

                        # if label == self.stopped:
                        #     self.stopcnt += 1
                        # elif label == self.forward:
                        #     self.fwdcnt += 1
                        # elif label == self.backward:
                        #     self.backcnt += 1
                        # elif label == self.left:
                        #     self.leftcnt += 1
                        # elif label == self.right:
                        #     self.rightcnt += 1
                        # elif label == self.Rright:
                        #     self.Rrightcnt += 1
                        # elif label == self.Rleft:
                        #     self.Rleftcnt += 1
                    except Exception as e:
                        pass

        np.random.shuffle(self.training_data)
        np.save("../training_data.npy", self.training_data)
        ###or return self.training_data

    def collect_data(self, img, motor, data_set='train'):
        save_path = os.path.join(LANE_ROOT, data_set)
        self.existing_dir(save_path)
        save_path = os.path.join(save_path, self.INV_LABELS[motor.drive_state])
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

def create_plot(name):
    style.use("ggplot")
    def create_acc_loss_graph(model_name):
        contents = open("../model.log", "r").read().split("\n")

        times = []
        accuracies = []
        losses = []

        val_accs = []
        val_losses = []

        for c in contents:
            if model_name in c:
                name, timestamp, acc, loss, val_acc, val_loss = c.split(",")

                times.append(float(timestamp))
                accuracies.append(float(acc))
                losses.append(float(loss))

                val_accs.append(float(val_acc))
                val_losses.append(float(val_loss))

        fig = plt.figure()

        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

        ax1.plot(times, accuracies, label="acc")
        ax1.plot(times, val_accs, label="val_acc")
        ax1.legend(loc=2)
        ax2.plot(times, losses, label="loss")
        ax2.plot(times, val_losses, label="val_loss")
        ax2.legend(loc=2)
        plt.show()
    create_acc_loss_graph(name)

def training_data_exists():
    if not os.path.exists('../training_data.npy'):
        dc = DataControl()
        dc.make_training_data()


