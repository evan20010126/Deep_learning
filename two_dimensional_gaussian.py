import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 定義參數
mu = np.array([2, 3])      # 平均值
cov_matrix = np.array([[1, 0.5], [0.5, 2]])  # 協方差矩陣

# 建立網格
x, y = np.meshgrid(np.linspace(-5, 8, 100), np.linspace(-5, 8, 100))
pos = np.dstack((x, y))

# 計算高斯分布
rv = multivariate_normal(mu, cov_matrix)
z = rv.pdf(pos)

# 繪製等高線圖
plt.contourf(x, y, z, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Gaussian Distribution')
plt.colorbar()
plt.show()
