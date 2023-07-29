import numpy as np
import torch

# def sigmoid(x):
#         return 1 / (1 + np.exp(-x))  

# print(sigmoid(np.array([0.1,0.2,0.3,0.4])))

# data = np.array([
#         [[1,1,1],[1,1,1],[1,1,1]],
#         [[2,2,2],[2,2,2],[2,2,2]],
#         [[3,3,3],[3,3,3],[3,3,3]]
# ])
# print(data.shape)
# print(np.dot(data, np.array([-1,-1,-1])))
# print(np.dot(data, np.array([-1,-1,-1])).shape)


# s = [1,2,3,4,5]
# for i in s[::-1]: 
#     print(i)

device = torch.device('cpu')
print(torch.cuda.is_available())

# x= np.random.rand(8,1)
# y=np.random.rand(1,1)
# print(np.dot(x,y).shape)

# x = [1]
# y= x.pop()
# print(y)


# x = np.array([0,-1,2,-3,4,5,6])
# x[x < 0] = 0
# print(x)