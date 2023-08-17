
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv(r"C:\Users\User\Desktop\Mono.csv", header=None).to_numpy()
start = 15
train_loss_mono = df[start:, 0]
valid_loss_mono = df[start:, 1]
# train_loss_cyclical = df[start:, 2]
valid_loss_cyclical = df[start:, 3]
# train_loss_without = df[start:, 4]
valid_loss_without = df[start:, 5]


# 輸入數據
epochs = range(start + 1,start + len(train_loss_mono) + 1)

# 設定圖表
plt.figure(figsize=(10, 6))
# plt.plot(epochs, train_loss_mono, label='Train Loss (Mono)', color='blue')
plt.plot(epochs, valid_loss_mono, label='Valid Loss (Mono)', color='orange')
# plt.plot(epochs, train_loss_cyclical,
        #  label='Train Loss (Cyclical)', color='green')
plt.plot(epochs, valid_loss_cyclical,
         label='Valid Loss (Cyclical)', color='red')
# plt.plot(epochs, train_loss_without,
        #  label='Train Loss (Without)', color='purple')
plt.plot(epochs, valid_loss_without,
         label='Valid Loss (Without)', color='brown')

# 添加標籤與標題
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# 顯示圖表
plt.show()
