import pandas as pd
import ResNet
import dataloader
import torch
import copy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda')
print(torch.cuda.is_available())

def test(model, test_dataloader):
    for (x_batch, y_batch) in test_dataloader:
        x_batch = x_batch.float().to(device)
        y_batch = y_batch.float().to(device)
        y_pred = model(x_batch)

        zero = torch.zeros_like(y_pred)
        one = torch.ones_like(y_pred)
        
        y_pred = torch.where(y_pred > 0.5, one, y_pred)
        y_pred = torch.where(y_pred < 0.5, zero, y_pred)
        acc = (y_pred == y_batch).sum().item()
        
        
        

    print("test() not defined")

best_model = None
all_activation_history = list()
def train(model, criterion, optimizer, train_dataloader, valid_loader, epochs, min_epoch):
    training_acc_history = list()
    training_loss_history = list()
    testing_acc_history = list()
    testing_loss_history = list()
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

        for batch_idx, (x_batch, y_batch) in enumerate(train_dataloader):
            # print("batch_idx:", batch_idx)
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            optimizer.zero_grad()

            y_pred = model(x_batch)
            
            # print(y_pred)
            # print(y_batch.shape)
            y_pred = y_pred.view(-1)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
            total_samples += y_batch.size(0)
            

            zero = torch.zeros_like(y_pred)
            one = torch.ones_like(y_pred)
            predicted = torch.where(y_pred > 0.5, one, y_pred)
            predicted = torch.where(y_pred < 0.5, zero, y_pred)
            correct_predictions += (predicted == y_pred).sum().item()

            batch_acc = (predicted == y_pred).sum().item() / 16.0

            
            if batch_idx%15 == 0:
                print('batch_idx', batch_idx, 'batch_acc', batch_acc)
            # _, predicted = torch.max(y_pred, 1)
            # _, true_predicted = torch.max(y_batch, 1)
            # correct_predictions += (predicted == true_predicted).sum().item()

        training_loss /= len(train_dataloader)
        accuarcy = correct_predictions / total_samples
        
        if epoch % 50 == 0:
            print(f'Train Epoch: {epoch + 1} Acc: {accuarcy}, Loss: {training_loss}')

        training_acc_history.append(accuarcy)
        training_loss_history.append(training_loss)

        testing_acc, testing_loss = evaluate(model, criterion, valid_loader)

        if  epoch > min_epoch and testing_loss <= min_loss_val:
            min_loss_val = testing_loss
            global best_model
            best_model = copy.deepcopy(model)
        
        testing_acc_history.append(testing_acc)
        testing_loss_history.append(testing_loss)
    all_activation_history.append([training_acc_history.copy(), testing_acc_history.copy(), training_loss_history.copy(), testing_loss_history.copy()])
    print(f"Last epoch: {all_activation_history[-1][0][-1], all_activation_history[-1][1][-1]} {all_activation_history[-1][2][-1]} {all_activation_history[-1][3][-1]}")

def evaluate(model, criterion, valid_dataloader):
    model.eval()

    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(valid_dataloader):
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().to(device)

            y_pred = model(x_batch)
            y_pred = y_pred.view(-1)
            test_loss += criterion(y_pred, y_batch).item()
            
            total_samples += y_batch.size(0)

            zero = torch.zeros_like(y_pred)
            one = torch.ones_like(y_pred)
            predicted = torch.where(y_pred > 0.5, one, y_pred)
            predicted = torch.where(y_pred < 0.5, zero, y_pred)
            correct_predictions += (predicted == y_pred).sum().item()
            # _, predicted = torch.max(y_pred, 1)
            # _, true_predicted = torch.max(y_batch, 1)
            # correct_predictions += (predicted == true_predicted).sum().item()

            # running_loss = 0.0    
    test_loss /= len(valid_dataloader) # 因為每個mini batch中在for loop已經 .item()加總
    accuracy = correct_predictions / total_samples
    # print(len(dataloader), total_samples)
    # print(f'Test Loss: {test_loss}, Test Accuracy: {accuracy}')
    return accuracy, test_loss

def plot_acc_loss(all_activation_history, epochs):
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


def save_result(csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv("./your_student_id_resnet18.csv", index=False)

if __name__ == "__main__":
    
    train_data = dataloader.LeukemiaLoader("new_dataset","train")
    valid_data = dataloader.LeukemiaLoader("new_dataset","valid")
    test_data = dataloader.LeukemiaLoader("new_dataset","test")

    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=None, shuffle=False)

    models = [
        ResNet.ResNet18((3, 450, 450)).to(device=device)
    ]
    training_paras = [
        # [lr, weight_decay, epochs, min_epoch]
        [1e-3, 0, 100, 10],
    ]
    criterion = torch.nn.functional.binary_cross_entropy
   

    for model, training_para in zip(models,training_paras):
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_para[0], weight_decay=training_para[1])
        train(model, criterion, optimizer, train_dataloader, valid_dataloader, epochs=training_para[2], min_epoch = training_para[3])
    
    model = best_model
    # testing_acc, testing_loss = testing(model, criterion, test_loader)
    # print(f"test acc: {testing_acc}, test loss: {testing_loss}")
    plot_acc_loss(all_activation_history)


    torch.save(model.state_dict(), "./models/resnet18.pth")
    print("Good Luck :)")

