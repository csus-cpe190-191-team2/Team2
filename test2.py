import convolutional_neural_network as cnn
import eyes
import torch
import transform as tf

if __name__ == '__main__':
    net = cnn.NeuralNet()
    #print('check0')
    data = torch.load('../catdognet.pt')
    #print('check1')
    net.load_state_dict(data)
    #print(net.state_dict())
    net = net.to(cnn.get_device())
    #net._model_name = cnn.get_name()
    #print('All good')
    img = eyes.load_img('../cattest3.jpg')
    net.predict(img)
    print('Done...')