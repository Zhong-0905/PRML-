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

#最小二乘法
#先在 X 矩阵中添加一列I构造增广矩阵
X_mat = np.column_stack((np.ones(len(X_train)), X_train))
Y_mat = y_train.reshape(-1, 1)
#beta=(X^T*X)^-1*X^T*Y
beta = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ Y_mat

intercept = beta[0][0] #截距
slope = beta[1][0] #斜率

print(f"拟合方程: y = {slope:.4f}x + {intercept:.4f}")

#预测与评估
y_pred = slope * X_test + intercept
mse = np.mean((y_test - y_pred)**2)
print(f"测试集均方误差: {mse:.4f}")

#可视化
plt.scatter(X_train, y_train, color='red', label='Training Data', alpha=0.5)
plt.scatter(X_test, y_test, color='blue', label='Test Data', alpha=0.5)
plt.plot(X_train, slope * X_train + intercept, color='green', linewidth=2, label='OLS Fit')
plt.legend()
plt.title("Least Squares Linear Regression") #最小二乘法线性拟合
plt.xlabel("x")
plt.ylabel("y")
plt.show()