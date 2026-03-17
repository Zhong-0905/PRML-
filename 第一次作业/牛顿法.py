import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#加载数据
file_path = 'Data4Regression.xlsx'
train_data = pd.read_excel(file_path, sheet_name=0) #表单一：训练集
test_data = pd.read_excel(file_path, sheet_name=1)  #表单二：测试集

#数据预处理，读取共4列，改为二维列向量
X_train = train_data.iloc[:, 0].values.reshape(-1, 1)
y_train = train_data.iloc[:, 1].values.reshape(-1, 1)
X_test = test_data.iloc[:, 0].values.reshape(-1, 1)
y_test = test_data.iloc[:, 1].values.reshape(-1, 1)

#牛顿法
#先在 X 矩阵中添加一列I构造增广矩阵
X_mat = np.column_stack((np.ones(len(X_train)), X_train))
n = len(X_train)

#参数设置
#theta = (截距 b 和斜率 w)
theta = np.zeros((2, 1)) 
epochs = 5
loss_history = []

#迭代
for i in range(epochs):
    #预测预测值
    y_pred = X_mat @ theta
    
    #计算MSE
    loss = np.mean((y_train - y_pred)**2)
    loss_history.append(loss)
    
    #计算一阶导(Gradient)
    gradient = (2/n) * (X_mat.T @ (X_mat @ theta - y_train))
    
    #计算二阶导(Hessian 矩阵)
    hessian = (2/n) * (X_mat.T @ X_mat)
    
    #更新参数
    theta = theta - np.linalg.inv(hessian) @ gradient
    
    print(f"Iteration {i+1}: Loss = {loss:.4f}, b = {theta[0][0]:.4f}, w = {theta[1][0]:.4f}")

#结果验证
b_final, w_final = theta[0][0], theta[1][0]
y_test_pred = w_final * X_test + b_final
mse_test = np.mean((y_test - y_test_pred)**2)

print(f"\n训练结束:")
print(f"拟合方程: y = {w_final:.4f}x + {b_final:.4f}")
print(f"测试集均方误差: {mse_test:.4f}")

#可视化
plt.figure(figsize=(12, 5))
#拟合效果
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='red', alpha=0.5, label='Train Data')
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Test Data')
x_range = np.linspace(X_train.min(), X_train.max(), 100)
y_range = w_final * x_range + b_final
plt.plot(x_range, y_range, color='green', linewidth=3, label='Newton Fit')
plt.title(f"Newton's Linear Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
#损失收敛曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), loss_history, 'o-', color='orange')
plt.title("Newton's Method Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.tight_layout()
plt.show()