import numpy as np
import matplotlib.pyplot as plt


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
    '''
    'FullyConnection'
    'Convolution'
    'Sigmoid'
    '''

    def __init__(self, input_shape, class_num):
        self.layers = list()
        # input shape = (2,1)
        '''
        'W'
        'Bias'
        'Sigmoid'
        '''
        layer0_unit_num = 4
        self.layers.append(['W', np.random.rand(layer0_unit_num, input_shape[0])])
        self.layers.append(['Bias', np.random.rand(layer0_unit_num,  1)])
        self.layers.append(['Sigmoid', None])

        layer1_unit_num = 8
        self.layers.append(['W', np.random.rand(layer1_unit_num, layer0_unit_num)])
        self.layers.append(['Bias', np.random.rand(layer1_unit_num, 1)])
        self.layers.append(['Sigmoid', None])
        self.layers.append(['W', np.random.rand(class_num, layer1_unit_num)])

        self.middle_output_stack = []
        
    def train(self, x, y, learning_rate, epoch):
        for i in range(epoch): 
            for j in range(x.shape[0]):
                tmp = x[j].reshape(-1, 1) # 2 -> 2x1
                pred_y = self.predict(tmp)
                print(f"epoch:{i} --> old loss: {(pred_y - y[j])**2}")
                self.backward(pred_y, y[j], learning_rate)

            
        

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
                grad = self.sigmoid_grad(grad)
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
        return x

    # def compute_Jacobian(self,  node_x, node_y):
    #     self.loss = 0
    #     Jacobian = np.zeros((node_y.shape[0], node_x.shape), float)
    #     '''
    #     Target:
    #     [
    #     [y1/x1 , y1/x2, y1/x3, ..., y1/xm],
    #     [y2/x1, y2/x2, y2/x3, ..., y2/xm],
    #     ...
    #     ]
    #     '''
    #     pass
    def sigmoid(self, x):
        return 1.0/ (1.0 + np.exp(-x))  
    
    def sigmoid_grad(self, x): # x 為 input sigmoid function 的參數
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)        


x, y = generate_XOR_easy()
x = x.reshape(-1, 2, 1)
y = y.reshape(-1, 1, 1)
model = my_model((2,1), 1)
model.train(x, y, 0.00001, 999)

acc = 0
for i in range(len(x)):
    # print(x.shape)
    y_pred = model.predict(x[i])
    # print(y.shape)
    # print(y_pred)
    ans = (y_pred[0][0] >= 0.5)
    print(ans)
    if ((y[i][0][0] and ans) or (y[i][0][0] == 0 and not ans)):
        acc += 1
print(acc/len(x))

for i in range(x.shape[0]):
    # print(y[i][0])
    if y[i][0] == 0:
        plt.scatter(x[i, 0], x[i, 1], color='hotpink')
    else:
        plt.scatter(x[i, 0], x[i, 1], color='green')
plt.show()
