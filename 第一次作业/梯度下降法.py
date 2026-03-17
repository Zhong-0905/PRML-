import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#加载数据
file_path = 'Data4Regression.xlsx'
train_data = pd.read_excel(file_path, sheet_name=0) #表单一：训练集
test_data = pd.read_excel(file_path, sheet_name=1)  #表单二：测试集

#数据预处理，读取共4列
X_train = train_data.iloc[:, 0].values
y_train = train_data.iloc[:, 1].values
X_test = test_data.iloc[:, 0].values
y_test = test_data.iloc[:, 1].values

#梯度下降参数设置
learning_rate = 0.01  #学习率
epochs = 2000         #迭代次数
w = 0.0               #初始斜率
b = 0.0               #初始截距
n = float(len(X_train))

#记录损失历史
loss_history = []

#迭代
for i in range(epochs):
    y_pred_current = w * X_train + b
    
    #计算MSE
    loss = (1/n) * sum((y_train - y_pred_current)**2)
    loss_history.append(loss)
    
    #计算梯度
    dw = (-2/n) * sum(X_train * (y_train - y_pred_current))
    db = (-2/n) * sum(y_train - y_pred_current)
    
    #更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    #200轮显示一下进度
    if i % 200 == 0:
        print(f"Epoch {i}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

#测试集评估
y_test_pred = w * X_test + b
mse_test = np.mean((y_test - y_test_pred)**2)

print(f"\n训练结束:")
print(f"拟合方程: y = {w:.4f}x + {b:.4f}")
print(f"测试集均方误差: {mse_test:.4f}")

#可视化
plt.figure(figsize=(12, 5))
#拟合效果
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='red', alpha=0.5, label='Train')
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Test')
plt.plot(X_train, w * X_train + b, color='green', linewidth=3, label='GD Fit')
plt.title("Gradient Descent Linear Fit")
plt.legend()
#损失下降曲线
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title("Loss Reduction over Epochs")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.show()