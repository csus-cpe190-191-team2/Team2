import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import transform as tf
from tqdm import tqdm
import os

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8)
        self.conv2 = nn.Conv2d(16, 32, 8)

        x = torch.randn(240, 480).view(-1, 1, 240, 480)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)  # flattening.
        self.fc2 = nn.Linear(512, 2)  # 512 in, 2 out bc we're doing 2 classes (dog vs cat).
        #self.fc2 = nn.Linear(512, 6)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

        self.VAL_PCT = 0.1  # lets reserve 10% of our data for validation
        self.val_size = 0

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            #print(self._to_linear)
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # bc this is our output layer. No activation here.
        return F.log_softmax(x, dim=1)

    def get_lin(self):
        return self._to_linear

    def save_tensors(self):
        training_data = np.load("../training_data.npy", allow_pickle=True)
        data_list = [i[0] for i in tqdm(training_data)]
        print('check0')
        data_list = np.array(data_list)
        print('check1')
        X = torch.Tensor(data_list).view(-1, 240, 480)
        #X = torch.as_tensor(data_list)
        print('check2')
        X = X / 255.0
        print('check3')
        torch.save(X, '../x_tensor.pt')
        data_list = [i[1] for i in tqdm(training_data)]
        print('check4')
        data_list = np.array(data_list)
        print('check5')
        y = torch.Tensor(data_list)
        #y = torch.as_tensor(data_list)
        torch.save(y, '../y_tensor.pt')
        self.val_size = int(len(X) * self.VAL_PCT)

    def load_tensors(self):
        X = torch.load('../x_tensor.pt')
        self.val_size = int(len(X) * self.VAL_PCT)
        return X, torch.load('../y_tensor.pt')

    def train_xy(self, X, y):
        return X[:-self.val_size], y[:-self.val_size]

    def test_xy(self, X, y):
        return X[-net.val_size:], y[-net.val_size:]

    def epochs(self, train_X, train_y, EPOCHS=1, BATCH_SIZE=100):
        print('epochs starting...')
        for epoch in tqdm(range(EPOCHS)):
            for i in tqdm(range(0, len(train_X),
                                BATCH_SIZE)):  # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
                # print(f"{i}:{i+BATCH_SIZE}")
                batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 240, 480)
                batch_y = train_y[i:i + BATCH_SIZE]

                self.zero_grad()

                outputs = self(batch_X)
                loss = self.loss_function(outputs, batch_y)
                loss.backward()
                self.optimizer.step()  # Does the update

            print(f"Epoch: {epoch}. Loss: {loss}")

    def test_acc(self, test_X, test_y):
        correct = 0
        total = 0
        with torch.no_grad():
            for i in tqdm(range(len(test_X))):
                real_class = torch.argmax(test_y[i])
                net_out = self(test_X[i].view(-1, 1, 240, 480))[0]  # returns a list,
                predicted_class = torch.argmax(net_out)

                if predicted_class == real_class:
                    correct += 1
                total += 1
        print("Accuracy: ", round(correct / total, 3))

if __name__ == '__main__':
    net = NeuralNet()
    print(net)

    if not os.path.exists('../training_data.npy'):
        dc = tf.DataControl()
        dc.make_training_data()

    if not os.path.exists('../x_tensor.pt'):
        net.save_tensors()
    X,y = net.load_tensors()
    print('Data transformed to tensors')

    train_X, train_y = net.train_xy(X,y)
    test_X, test_y = net.test_xy(X,y)
    print(len(train_X), len(test_X))

    net.epochs(train_X, train_y)

    net.test_acc(test_X, test_y)
