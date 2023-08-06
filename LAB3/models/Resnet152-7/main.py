import pandas as pd
import ResNet
import dataloader
import torch
import copy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
import openai
import numpy as np
import shutil
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
print(torch.cuda.device_count())
torch.cuda.empty_cache() 
# Move your model and data to the GPU

def comapre_ans(y_pred, y_groundTruth, mode):
    if mode == 'probabiliy_sigmoid':
        zero = torch.zeros_like(y_pred)
        one = torch.ones_like(y_pred)
        y_pred = torch.where(y_pred > 0.5, one, y_pred)
        y_pred = torch.where(y_pred < 0.5, zero, y_pred)
        correct_predictions = (y_pred == y_groundTruth).sum().item()
    
    return correct_predictions 

def test(model, test_dataloader):
    acc = 0
    for (x_batch, y_batch) in test_dataloader:
        x_batch = x_batch.float().to(device)
        y_batch = y_batch.float().to(device)
        y_pred = model(x_batch)

        y_pred = y_pred.view(-1)            
        acc =  comapre_ans(y_pred, y_batch, 'probabiliy_sigmoid') / y_batch.size(dim = 0)

    return acc
        

best_model = None
done_epoch = 0
all_activation_history = list()
training_acc_history = list()
training_loss_history = list()
valid_acc_history = list()
valid_loss_history = list()
def train(model, criterion, optimizer, train_dataloader, valid_loader, epochs, min_epoch):
    training_acc_history.clear()
    training_loss_history.clear()
    valid_acc_history.clear()
    valid_loss_history.clear()
    min_loss_val = 10
    
    
    for epoch in range(epochs):
        model.train()

        training_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_dataloader):
            # print("batch_idx:", batch_idx)
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            y_pred = y_pred.view(-1)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
            total_samples += y_batch.size(0)
            
            tmp = comapre_ans(y_pred, y_batch, 'probabiliy_sigmoid')
            correct_predictions += tmp

            if batch_idx%15 == 0:
                print(f"BATCH EPOCH: {batch_idx}, 'BATCH ACC: {tmp/y_batch.size(0)}")
            # batch_acc = (predicted == y_pred).sum().item() / 16.0
            # if batch_idx%15 == 0:
            #     print('batch_idx', batch_idx, 'batch_acc', batch_acc)

            # _, predicted = torch.max(y_pred, 1)
            # _, true_predicted = torch.max(y_batch, 1)
            # correct_predictions += (predicted == true_predicted).sum().item()

        training_loss /= len(train_dataloader)
        accuarcy = correct_predictions / total_samples
        
        if epoch % 1 == 0:
            print(f'Train Epoch: {epoch + 1} Acc: {accuarcy}, Loss: {training_loss}')

        training_acc_history.append(accuarcy)
        training_loss_history.append(training_loss)

        valid_acc, valid_loss = evaluate(model, criterion, valid_loader)
        print(f'Val acc: {valid_acc}, Val Loss: {valid_loss}')

        if  epoch > min_epoch and valid_loss <= min_loss_val:
            print("update best model")
            min_loss_val = valid_loss
            global best_model
            best_model = copy.deepcopy(model)
        
        valid_acc_history.append(valid_acc)
        valid_loss_history.append(valid_loss)
        global done_epoch
        done_epoch = epoch
    all_activation_history.append([training_acc_history.copy(), valid_acc_history.copy(), training_loss_history.copy(), valid_loss_history.copy()])
    print(f"Last epoch: {all_activation_history[-1][0][-1], all_activation_history[-1][1][-1]} {all_activation_history[-1][2][-1]} {all_activation_history[-1][3][-1]}")

def evaluate(model, criterion, valid_dataloader):
    model.eval()

    valid_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(valid_dataloader):
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)

            y_pred = model(x_batch)
            y_pred = y_pred.view(-1)
            valid_loss += criterion(y_pred, y_batch).item()
            
            total_samples += y_batch.size(0)
            correct_predictions +=  comapre_ans(y_pred, y_batch, 'probabiliy_sigmoid')
            # _, predicted = torch.max(y_pred, 1)
            # _, true_predicted = torch.max(y_batch, 1)
            # correct_predictions += (predicted == true_predicted).sum().item()

    valid_loss /= len(valid_dataloader) # 因為每個mini batch中在for loop已經 .item()加總
    accuracy = correct_predictions / total_samples
    # print(len(dataloader), total_samples)
    # print(f'Test Loss: {valid_loss}, Test Accuracy: {accuracy}')
    return accuracy, valid_loss

def plot_acc_loss(all_activation_history, epochs, save_path):
    colors = [['red', 'orange'], ['blue', 'cyan'], ['green', 'purple']]
    labels = [['resnet18_train', 'resnet18_valid'], ['leaky_relu_train', 'leaky_relu_test'], ['elu_train', 'elu_test']]
    fig, axs = plt.subplots(2, 1)
    for color_idx, (training_acc_history, valid_acc_history, training_loss_history, valid_loss_history) in enumerate(all_activation_history):
        axs[0].plot(range(1, epochs + 1), training_acc_history, color=colors[color_idx][0], label=labels[color_idx][0] + '_acc', linestyle='-')
        axs[0].plot(range(1, epochs + 1), valid_acc_history, color=colors[color_idx][1], label=labels[color_idx][1] + '_acc', linestyle='-')

        axs[1].plot(range(1, epochs + 1), training_loss_history, color=colors[color_idx][0], label=labels[color_idx][0] + '_loss', linestyle='-')
        axs[1].plot(range(1, epochs + 1), valid_loss_history, color=colors[color_idx][1], label=labels[color_idx][1] + '_loss', linestyle='-')
    axs[0].set_title('Accuracy curves')
    axs[1].set_title('Loss curves')
    axs[0].set_ylabel('Accuracy')
    axs[1].set_ylabel('Loss')
    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f"./models/{save_path}/learning_curve.jpg")
    plt.show()

def save_result(csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv("./your_student_id_resnet18.csv", index=False)

if __name__ == "__main__":

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation([0, 15]),
        transforms.RandomRotation([90, 180]),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    
    train_data = dataloader.LeukemiaLoader("new_dataset","train", transform = transform_train)
    valid_data = dataloader.LeukemiaLoader("new_dataset","valid", transform = transform_test) # convert type
    test_data = dataloader.LeukemiaLoader("new_dataset","test",  transform = transform_test) # convert type
    

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=8, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle=False)
    # plt.imshow(torch.transpose(next(iter(train_dataloader))[0][0], 0 , 2).numpy())
    # plt.show()
    # exit(0)

    models = [
        # openai.ResNet18(1).to(device=device)
        # ResNet.ResNet18((3, None, None), filters=8).to(device=device),
        # ResNet.ResNet50((3, None, None), filters=2).to(device=device),
        ResNet.ResNet152((3, None, None), filters=24).to(device=device)
    ]
    training_paras = [
        # [lr, weight_decay, epochs, min_epoch]
        [1e-4, 0, 20, 0],
    ]

    save_paths = [
        'Resnet152-7'
    ]

    criterion = torch.nn.functional.binary_cross_entropy
   
   # -- train --
    # '''
    for model, training_para, save_path in zip(models,training_paras,save_paths):
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_para[0], weight_decay=training_para[1])
        try:
            train(model, criterion, optimizer, train_dataloader, valid_dataloader, epochs=training_para[2], min_epoch = training_para[3])
            model = best_model
            shutil.copy("main.py", f"./models/{save_path}/main.py")
            torch.save(model.state_dict(), f"./models/{save_path}/resnet18.pth")
            # testing_acc, testing_loss = testing(model, criterion, test_loader)
            # print(f"test acc: {testing_acc}, test loss: {testing_loss}")
            # plot_acc_loss(all_activation_history, epochs=training_paras[0][2], save_path=save_path)
            plot_acc_loss(all_activation_history, epochs=training_paras[0][2], save_path=save_path)
        except:
            model = best_model
            os.makedirs(f"./models/{save_path}")
            shutil.copy("main.py", f"./models/{save_path}/main.py")
            torch.save(model.state_dict(), f"./models/{save_path}/resnet18.pth")
            all_activation_history.append([training_acc_history.copy(), valid_acc_history.copy(), training_loss_history.copy(), valid_loss_history.copy()])
            plot_acc_loss(all_activation_history, epochs=done_epoch+1, save_path=save_path)
            exit(0)
    # '''
    # -- test --
    '''
    model = models[0]
    model.load_state_dict(torch.load("./models/Resnet152-5/resnet18.pth"))

    predict_result = np.array([])
    for batch_idx, (x_batch) in enumerate(test_dataloader):
        x_batch = x_batch.float().to(device)

        y_pred = model(x_batch)
        y_pred = y_pred.view(-1)
        
        zero = torch.zeros_like(y_pred)
        one = torch.ones_like(y_pred)
        y_pred = torch.where(y_pred > 0.5, one, y_pred)
        y_pred = torch.where(y_pred < 0.5, zero, y_pred)
        predict_result = np.append(predict_result, y_pred.cpu().detach().numpy())
    save_result("resnet_152_test.csv", predict_result)
    '''

    print("Good Luck :)")
    

