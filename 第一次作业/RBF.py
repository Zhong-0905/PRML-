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

#构造RBF特征
centers = np.linspace(0, 10, 12) #取10个中心点
sigma = 1.0 # 控制波浪的宽度

def r2f_transform(X, centers, sigma):
    return np.exp(-(X[:, np.newaxis] - centers)**2 / (2 * sigma**2))

X_rbf_train = r2f_transform(X_train, centers, sigma)
X_rbf_test = r2f_transform(X_test, centers, sigma)

#加上偏置项(截距)
X_rbf_train = np.column_stack((np.ones(len(X_rbf_train)), X_rbf_train))
X_rbf_test = np.column_stack((np.ones(len(X_rbf_test)), X_rbf_test))

# 最小二乘法求解
beta = np.linalg.inv(X_rbf_train.T @ X_rbf_train) @ X_rbf_train.T @ y_train

#预测
y_pred = X_rbf_test @ beta
mse = np.mean((y_test - y_pred)**2)
print(f"测试集均方误差: {mse:.4f}")


#可视化
plt.scatter(X_train, y_train, color='red', label='Training Data', alpha=0.5)
plt.scatter(X_test, y_test, color='blue', label='Test Data', alpha=0.5)
x_smooth = np.linspace(0, 10, 500)
X_smooth_rbf = r2f_transform(x_smooth, centers, sigma)
X_smooth_rbf = np.column_stack((np.ones(len(X_smooth_rbf)), X_smooth_rbf))
y_smooth = X_smooth_rbf @ beta
plt.plot(x_smooth, y_smooth, color='green', linewidth=2, label='RBF Fit')
plt.title("Non-linear Fit using Radial Basis Functions")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()