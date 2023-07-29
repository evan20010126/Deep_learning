import numpy as np
import matplotlib.pyplot as plt
import json


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


class my_model():
    def __init__(self, input_shape, class_num):
        self.layers = list()
        '''
        # input shape = (2, batch_size)
        '['W', np.array]'
        '['Bias', np.array]'
        '['Sigmoid', np.array([None])]'
        '['ReLU', np.array([None])'
        '''
        layer0_unit_num = 4
        BATCH_SIZE = input_shape[1]
        self.layers.append(['W', np.random.rand(layer0_unit_num, input_shape[0])])
        self.layers.append(['Bias', np.random.rand(layer0_unit_num,  BATCH_SIZE)])
        self.layers.append(['ReLU', np.array([None])])

        layer1_unit_num = 2
        self.layers.append(['W', np.random.rand(layer1_unit_num, layer0_unit_num)])
        self.layers.append(['Bias', np.random.rand(layer1_unit_num, BATCH_SIZE)])
        self.layers.append(['ReLU', np.array([None])])
        self.layers.append(['W', np.random.rand(class_num, layer1_unit_num)])
        self.layers.append(['Sigmoid', np.array([None])])

        self.middle_output_stack = []
        self.learning_history = []
        
    def train(self, x, y, learning_rate, epoch):
        for i in range(epoch):
            pred_y = self.predict(x)
            loss = np.sum((pred_y - y)**2) / y.shape[-1]
            if (i % 5000) == 0 or (i == epoch-1):            
                print(f"epoch[{i:6}] loss: {loss}")
            self.learning_history.append([i, loss])
            self.backward(pred_y, y, learning_rate)


    def backward(self, pred_y, y, learning_rate):
        # compute gradient, update parameter
        loss = (pred_y - y)
        
        grad = 2.0 *  loss # loss derived by pred_y
        for neg_idx, layer in enumerate(self.layers[::-1], start=0):
            layerName, layerParameter = layer
            layer_num = len(self.layers) - neg_idx - 1 # for update parameter

            middle_output = self.middle_output_stack.pop()
            if (layerName == 'W'):
                # print(grad.shape)
                self.layers[layer_num][1] -= learning_rate * np.dot(grad, middle_output.T)
                grad = np.dot(layerParameter.T, grad)
            elif (layerName == 'Bias'):
                self.layers[layer_num][1] -= learning_rate * (grad * 1)
                grad = grad * 1
            elif (layerName == 'Sigmoid'):
                grad = self.sigmoid_grad(middle_output) * grad
                # no update parameter but transmit loss to lower layer.
            elif (layerName == 'ReLU'):
                grad = self.relu_grad(middle_output, grad)
                # no update parameter but transmit loss to lower layer.

    def predict(self, x):
        self.middle_output_stack.clear()
        for layer in self.layers:
            layerName, layerParameter = layer
            self.middle_output_stack.append(x)
            if (layerName == 'W'):
                x = np.dot(layerParameter, x)
            elif (layerName == 'Bias'):
                x = x + layerParameter
            elif (layerName == 'Sigmoid'):
                x = self.sigmoid(x=x)
            elif (layerName == 'ReLU'):
                x = self.relu(x=x)
        return x
    
    def save_weight(self, path):
        obj = []
        for layer in self.layers:
            layerName, layerParameter = layer
            obj.append([layerName, layerParameter.tolist()])

        with open(path, 'w') as f:
            json.dump({'myparameter':obj, 'learning_history': self.learning_history}, f)
        
    def load_weight(self, path):
        self.layers.clear()
        obj = None
        with open(path, 'r') as f:
            obj = json.load(f)
        
        self.learning_history = obj['learning_history']
        obj = obj['myparameter']
        for layer in obj:
            layerName, layerParameter = layer
            self.layers.append([layerName, np.array(layerParameter)])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  
    
    def sigmoid_grad(self, x):
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)
    
    def relu(self, x):
        x[x < 0] = 0
        return x
    
    def relu_grad(self, x, grad):
        grad[x < 0] = 0
        return grad

def show_each_prediction_returnACC(y, y_pred, BATCH_SIZE):
    bool_y_pred = (y_pred >= 0.5)
    bool_y = (y >= 0.5)
    acc = (bool_y_pred == bool_y).sum() / BATCH_SIZE
    for i in range(y.shape[-1]):
        print(f"Iter{i:2} | Ground truth: {y[0][i]} | Prediction: {y_pred[0][i]:.5f}[{bool_y_pred[0][i]:1}]")
    
    loss = np.sum((y - y_pred) ** 2) / BATCH_SIZE
    return acc, loss

def show_plot(x, y, y_pred):
    bool_y_pred = (y_pred >= 0.5)
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[1]):
        if y[0][i] == 0:
            plt.plot(x[0][i], x[1][i], 'ro')
        else:
            plt.plot(x[0][i], x[1][i], 'bo')
    
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[1]):
        if not bool_y_pred[0][i]:
            plt.plot(x[0][i], x[1][i], 'ro')
        else:
            plt.plot(x[0][i], x[1][i], 'bo')
    plt.show()

def show_learning_curve(history):
    history = np.array(history)
    plt.plot(history[:, 0], history[:, 1])
    plt.title("epoch/loss learning curve")
    plt.show()

def main_code(gernerate_func):
    x, y = gernerate_func()
    print(x.shape) # (21,2)
    print(y.shape)
    x = x.T # shape: (points, batch_size) (2,21)
    y = y.reshape(1, -1)
    BATCH_SIZE = y.shape[-1]
    model = my_model((2, BATCH_SIZE), 1)
    model.train(x, y, 1e-4, 50000)
    model.save_weight(path = './Deep_learning/weight')
    # model.load_weight(path = './Deep_learning/weight')
    y_pred = model.predict(x)
    
    acc, loss = show_each_prediction_returnACC(y, y_pred, BATCH_SIZE)
    print(f"Acc: {acc:.2%} | loss: {loss}")
    show_learning_curve(model.learning_history)
    show_plot(x, y, y_pred)

main_code(generate_linear)