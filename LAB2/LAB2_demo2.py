import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import dataloader as dl
import numpy as np
# from torchsummary import summary
import matplotlib.pyplot as plt
import copy

all_activation_history = list()

device = torch.device('cuda')
print(torch.cuda.is_available())

train_data, train_label, test_data, test_label = dl.read_bci_data()

# one-hot label
print(np.unique(train_label))
train_label = train_label.astype(int)
train_label = np.eye(2)[train_label] # one-hot
test_label = test_label.astype(int)
test_label = np.eye(2)[test_label] # one-hot

# print(train_label)
print(train_data.shape, train_label.shape)

loader = DataLoader(TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label)), batch_size=16, shuffle=True)

test_loader = DataLoader(TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label)), batch_size=16, shuffle=False)

class EEG_net(torch.nn.Module):
    def __init__(self, C, T, F1 = 32, D = 2, F2 = 64, N = 2, activation_layer = torch.nn.ELU(alpha=1.0)):
        # 論文版本
        super(EEG_net, self).__init__()
        # Conv2d
        self.conv_0 = torch.nn.Conv2d(1, F1, kernel_size=(1,64), stride=1, padding='same', bias=False)
        self.conv_1 = torch.nn.BatchNorm2d(F1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # DepthwiseConv2D
        self.depthwise_0 = torch.nn.Conv2d(F1, D * F1, kernel_size=(C,1), stride=1, groups=F1, bias=False)
        self.depthwise_1 = torch.nn.BatchNorm2d(F1 * D, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.depthwise_2 = activation_layer
        # self.depthwise_3 = torch.nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0)
        self.depthwise_3 = torch.nn.AvgPool2d(kernel_size=(1,4))
        self.depthwise_4 = torch.nn.Dropout(p=0.25) 
        # SeparableConv2D
        self.separable_0 = torch.nn.Conv2d(F1 * D, F2, kernel_size=(1,16), stride=1, padding='same', bias=False)
        self.separable_1 = torch.nn.BatchNorm2d(F2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.separable_2 = activation_layer
        # self.separable_3 = torch.nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0)
        self.separable_3 = torch.nn.AvgPool2d(kernel_size=(1,8))
        self.separable_4 = torch.nn.Dropout(p=0.5) 
        # Classification
        self.fatten = torch.nn.Flatten()
        self.classification_0 = torch.nn.Linear(in_features=(F2 * (T // 32)), out_features=N, bias=True)
        self.classification_1 = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.depthwise_0(x)
        x = self.depthwise_1(x)
        x = self.depthwise_2(x)
        x = self.depthwise_3(x)
        x = self.depthwise_4(x)
        x = self.separable_0(x)
        x = self.separable_1(x)
        x = self.separable_2(x)
        x = self.separable_3(x)
        x = self.separable_4(x)
        x = self.fatten(x)
        x = self.classification_0(x)
        x = self.classification_1(x)
        return x


class DeepConvNet(torch.nn.Module):
    def __init__(self, C, N, filters = 25, activation_layer = torch.nn.ELU(alpha=1.0)):
        super(DeepConvNet, self).__init__()
        self.firstPart = torch.nn.Sequential(
            torch.nn.Conv2d(1, filters, kernel_size=(1,5), stride=1, padding='valid', bias=False),
            torch.nn.Conv2d(filters, filters, kernel_size=(C,1), stride=1, padding='valid', bias=False),
            torch.nn.BatchNorm2d(filters, eps=1e-05, momentum=0.1),
            activation_layer,
            torch.nn.MaxPool2d((1,2)),
            torch.nn.Dropout(p=0.25)
        )
        self.secondPart = torch.nn.Sequential(
            torch.nn.Conv2d(filters, filters * 2, kernel_size=(1,5), stride=1, padding='valid', bias=False),
            torch.nn.BatchNorm2d(filters * 2, eps=1e-05, momentum=0.1),
            activation_layer,
            torch.nn.MaxPool2d((1,2)),
            torch.nn.Dropout(p=0.25)
        )
        self.thirdPart = torch.nn.Sequential(
            torch.nn.Conv2d(filters * 2, filters * 4, kernel_size=(1,5), stride=1, padding='valid', bias=False),
            torch.nn.BatchNorm2d(filters * 4, eps=1e-05, momentum=0.1),
            activation_layer,
            torch.nn.MaxPool2d((1,2)),
            torch.nn.Dropout(p=0.75)
        )
        self.forthPart = torch.nn.Sequential(
            torch.nn.Conv2d(filters * 4, filters * 8, kernel_size=(1,5), stride=1, padding='valid', bias=False),
            torch.nn.BatchNorm2d(filters * 8, eps=1e-05, momentum=0.1),
            activation_layer,
            torch.nn.MaxPool2d((1,2)),
            torch.nn.Dropout(p=0.75)
        )
        self.Flatten = torch.nn.Flatten()
        self.Dense = torch.nn.Linear(in_features= 15480, out_features=N, bias=True) # 8600
        self.Softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.firstPart(x)
        x = self.secondPart(x)
        x = self.thirdPart(x)
        x = self.forthPart(x)
        x = self.Flatten(x)
        x = self.Dense(x)
        x = self.Softmax(x)
        return x

training_acc_history = list()
training_loss_history = list()
testing_acc_history = list()
testing_loss_history = list()
best_model = None

def training(model, criterion, optimizer, dataloader, epochs, min_epoch):
    training_acc_history.clear()
    training_loss_history.clear()
    testing_acc_history.clear()
    testing_loss_history.clear()
    min_loss_val = 10
    
    for epoch in range(epochs):
        model.train()

        training_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            optimizer.zero_grad()

            y_pred = model(x_batch)
            
            # print(y_pred)
            # print(y_batch.shape)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
            total_samples += y_batch.size(0)
            
            _, predicted = torch.max(y_pred, 1)
            _, true_predicted = torch.max(y_batch, 1)
            correct_predictions += (predicted == true_predicted).sum().item()

        training_loss /= len(dataloader)
        accuarcy = correct_predictions / total_samples
        
        if epoch % 50 == 0:
            print(f'Train Epoch: {epoch + 1} Acc: {accuarcy}, Loss: {training_loss}')

        training_acc_history.append(accuarcy)
        training_loss_history.append(training_loss)

        testing_acc, testing_loss = testing(model, criterion, test_loader)

        if  epoch > min_epoch and testing_loss <= min_loss_val:
            min_loss_val = testing_loss
            global best_model
            best_model = copy.deepcopy(model)
        
        testing_acc_history.append(testing_acc)
        testing_loss_history.append(testing_loss)
    all_activation_history.append([training_acc_history.copy(), testing_acc_history.copy(), training_loss_history.copy(), testing_loss_history.copy()])
    print(f"Last epoch: {all_activation_history[-1][0][-1], all_activation_history[-1][1][-1]} {all_activation_history[-1][2][-1]} {all_activation_history[-1][3][-1]}")
    
    # plt.plot(range(1, epochs + 1), training_acc_history, color = 'red')
    # plt.plot(range(1, epochs + 1), testing_acc_history, color = 'blue')
    # plt.show()
    # plt.plot(range(1, epochs + 1), training_loss_history, color = 'red')
    # plt.plot(range(1, epochs + 1), testing_loss_history, color = 'blue')
    # plt.show()

def testing(model, criterion, dataloader):
    model.eval()

    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)

            y_pred = model(x_batch)
            test_loss += criterion(y_pred, y_batch).item()
            
            total_samples += y_batch.size(0)

            _, predicted = torch.max(y_pred, 1)
            _, true_predicted = torch.max(y_batch, 1)
            correct_predictions += (predicted == true_predicted).sum().item()

            # running_loss = 0.0    
    test_loss /= len(dataloader) # 因為每個mini batch中在for loop已經 .item()加總
    accuracy = correct_predictions / total_samples
    # print(len(dataloader), total_samples)
    # print(f'Test Loss: {test_loss}, Test Accuracy: {accuracy}')
    return accuracy, test_loss


models = [
    # EEG_net(C = 2, T = 750, activation_layer=torch.nn.ReLU()).to(device=device),
    # EEG_net(C = 2, T = 750, activation_layer=torch.nn.LeakyReLU()).to(device=device), # the best
    # EEG_net(C = 2, T = 750).to(device=device), # 2 channels and 750 times
    # DeepConvNet(C = 2, N = 2, filters=45, activation_layer=torch.nn.ReLU()).to(device=device),
    DeepConvNet(C = 2, N = 2, filters=45, activation_layer=torch.nn.LeakyReLU()).to(device=device),
    # DeepConvNet(C = 2, N = 2, filters=45).to(device=device),
]

'''
for model in models:
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=7e-1)
    epochs = 500
    training(model, criterion, optimizer, loader, epochs=epochs, min_epoch = 200)

colors = [['red', 'orange'], ['blue', 'cyan'], ['green', 'purple']]
labels = [['relu_train', 'relu_test'], ['leaky_relu_train', 'leaky_relu_test'], ['elu_train', 'elu_test']]
fig, axs = plt.subplots(2, 1)
for color_idx, (training_acc_history, testing_acc_history, training_loss_history, testing_loss_history) in enumerate(all_activation_history):
    axs[0].plot(range(1, epochs + 1), training_acc_history, color=colors[color_idx][0], label=labels[color_idx][0] + '_acc', linestyle='-')
    axs[0].plot(range(1, epochs + 1), testing_acc_history, color=colors[color_idx][1], label=labels[color_idx][1] + '_acc', linestyle='-')

    axs[1].plot(range(1, epochs + 1), training_loss_history, color=colors[color_idx][0], label=labels[color_idx][0] + '_loss', linestyle='-')
    axs[1].plot(range(1, epochs + 1), testing_loss_history, color=colors[color_idx][1], label=labels[color_idx][1] + '_loss', linestyle='-')
axs[0].set_title('Accuracy curves')
axs[1].set_title('Loss curves')
axs[0].set_ylabel('Accuracy')
axs[1].set_ylabel('Loss')
axs[0].legend(loc='upper left')
axs[1].legend(loc='upper left')
plt.tight_layout()
plt.savefig('new.jpg')
plt.show()

# # SAVE ------------------------------------------------------------------------------------
model = best_model
testing_acc, testing_loss = testing(model, criterion, test_loader)
print(f"test acc: {testing_acc}, test loss: {testing_loss}")
torch.save(model.state_dict(), "best_model_parameters")

'''


# # LOAD -----------------------------------------------------------------------------------
best_leaky_eeg = DeepConvNet(C = 2, N = 2, filters=45, activation_layer=torch.nn.LeakyReLU()).to(device=device)
best_leaky_eeg.load_state_dict(torch.load('best_model_parameters_demo2'))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(best_leaky_eeg.parameters(), lr=1e-3, weight_decay=1e-3)
epochs = 200 * 5 
testing_acc, testing_loss = testing(best_leaky_eeg, criterion, test_loader)
print(f"test acc: {testing_acc}, test loss: {testing_loss}")
# '''
