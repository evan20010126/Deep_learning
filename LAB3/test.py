import torch
import torch.nn as nn

# 假設您有一個模型 model 和一個損失函數 criterion
model = nn.Linear(10, 1)
criterion = nn.MSELoss()

# 假設您有一組輸入 input 和對應的目標值 target
input = torch.randn(1, 10)
target = torch.randn(1)

# 前向傳遞，計算預測值
output = model(input)

# 計算損失
loss = criterion(output, target)

# 獲取損失的數值
loss_value = loss.item()

print(loss_value)
