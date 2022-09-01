import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import transform as tf
from tqdm import tqdm
import os
import time

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
        self._to_device = None
        self._model_name = ''

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

    def save_tensors(self, x_name, y_name):
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
        torch.save(X, x_name)
        data_list = [i[1] for i in tqdm(training_data)]
        print('check4')
        data_list = np.array(data_list)
        print('check5')
        y = torch.Tensor(data_list)
        #y = torch.as_tensor(data_list)
        torch.save(y, y_name)

    def load_tensors(self):
        X = torch.load('../x_tensor.pt')
        self.val_size = int(len(X) * self.VAL_PCT)
        return X, torch.load('../y_tensor.pt')

    def train_xy(self, X, y):
        return X[:-self.val_size], y[:-self.val_size]

    def test_xy(self, X, y):
        return X[-net.val_size:], y[-net.val_size:]

    def train(self, train_X, train_y, test_X, test_y, EPOCHS=1, BATCH_SIZE=100, to_test=False):
        print('epochs starting...')
        with open("../model.log", "a") as f:
            for epoch in range(EPOCHS):
                for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                    batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 240, 480)
                    batch_y = train_y[i:i + BATCH_SIZE]

                    batch_X, batch_y = batch_X.to(self._to_device), batch_y.to(self._to_device)

                    acc, loss = self.fwd_pass(batch_X, batch_y, train=True)

                    # print(f"Acc: {round(float(acc),2)}  Loss: {round(float(loss),4)}")
                    if to_test:
                        if i % 50 == 0:
                            val_acc, val_loss = self.quick_test(test_X, test_y, size=100)
                            f.write(
                                f"{self._model_name},{round(time.time(), 3)},{round(float(acc), 2)},{round(float(loss), 4)},{round(float(val_acc), 2)},{round(float(val_loss), 4)}\n")

    def quick_test(self, test_X, test_y, size=32):
        X, y = test_X[:size], test_y[:size]
        val_acc, val_loss = self.fwd_pass(X.view(-1, 1, 240, 480).to(self._to_device), y.to(self._to_device))
        return val_acc, val_loss

    def test_acc(self, test_X, test_y):  ###deprecated but still works
        correct = 0
        total = 0
        with torch.no_grad():
            for i in tqdm(range(len(test_X))):
                real_class = torch.argmax(test_y[i]).to(self._to_device)
                net_out = self(test_X[i].view(-1, 1, 240, 480).to(self._to_device))[0]  # returns a list of probabilities,
                predicted_class = torch.argmax(net_out)

                if predicted_class == real_class:
                    correct += 1
                total += 1
        print("Accuracy: ", round(correct / total, 3))

    def test_batch_acc(self, test_X, test_y, BATCH_SIZE=100):  ###deprecated but still works
        for i in tqdm(range(0, len(test_X), BATCH_SIZE)):

            batch_X = test_X[i:i + BATCH_SIZE].view(-1, 1, 240, 480).to(self._to_device)
            batch_y = test_y[i:i + BATCH_SIZE].to(self._to_device)

            acc, loss = self.fwd_pass(batch_X, batch_y)

            print(f"Acc: {round(float(acc), 2)}  Loss: {round(float(loss), 4)}")

    def fwd_pass(self, X, y, train=False):

        if train:
            self.zero_grad()
        outputs = self(X)
        matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
        acc = matches.count(True) / len(matches)
        loss = self.loss_function(outputs, y)

        if train:
            loss.backward()
            self.optimizer.step()

        return acc, loss

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def predict(self, img):
        input = torch.Tensor(img).view(-1, 1, 240, 480).to(get_device())
        output = self(input)
        print(output)
        prediction = torch.argmax(output)
        #print(prediction)
        print(f'Prediction is {tf.DataControl.INV_CD_LABELS[prediction.item()]}')

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def get_name():
    return f"model-{int(time.time())}"

def setup_net():  #only for creating new models
    net = NeuralNet().to(get_device())
    net._to_device = get_device()
    net._model_name = get_name()
    return net

def tensors_exist(net, name_x, name_y):
    if not os.path.exists(name_y):
        net.save_tensors(name_x, name_y)

def load_model(path):
    if os.path.exists(path):
        return torch.load(path)
    else:
        print('Model does not exist')

if __name__ == '__main__':
    net = setup_net()
    print(net)

    tf.training_data_exists()

    tensors_exist(net, '../x_tensor.pt', '../y_tensor.pt')
    X,y = net.load_tensors()
    print('Data transformed to tensors')

    train_X, train_y = net.train_xy(X,y) #returns part of tensor
    test_X, test_y = net.test_xy(X,y)    #returns other part of tensor
    print(len(train_X), len(test_X))

    net.train(train_X, train_y, test_X, test_y, EPOCHS=5,to_test=True)

    net.save_model('../catdognet.pt')

    #net.test_acc(test_X, test_y)

    tf.create_plot(net._model_name)


