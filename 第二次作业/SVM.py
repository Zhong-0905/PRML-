import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
# 3D Make Moons 数据生成函数
def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)
    
    # 构建两类数据 (C0 和 C1)
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # 加入高斯噪声
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y

# 生成训练集 (1000个点) 和测试集 (500个点)
X_train, y_train = make_moons_3d(n_samples=500, noise=0.1)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.1)

# SVM 三种核函数定义
kernels = {
    "SVM (Linear)": SVC(kernel='linear', C=1.0),
    "SVM (Poly)": SVC(kernel='poly', degree=3, C=1.0, coef0=1),
    "SVM (RBF)": SVC(kernel='rbf', gamma='auto', C=1.0)
}

results = {}

# 训练与评估
print(f"{'Kernel':<15} | {'Train Acc':<10} | {'Test Acc':<10}")
print("-" * 45)

for name, clf in kernels.items():
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    results[name] = test_acc
    print(f"{name:<15} | {train_acc*100:>8.2f}% | {test_acc*100:>8.2f}%")

# 可视化
fig = plt.figure(figsize=(18, 6))

for i, (name, clf) in enumerate(kernels.items()):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    y_pred = clf.predict(X_test)
    
    # 绘制真实点云
    ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], X_test[y_test==0, 2], 
               c='#ADD8E6', alpha=0.4, s=15)
    ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], X_test[y_test==1, 2], 
               c='#FFB6C1', alpha=0.4, s=15)
    
    # 绘制误分类点
    errors = (y_test != y_pred)
    if np.any(errors):
        ax.scatter(X_test[errors, 0], X_test[errors, 1], X_test[errors, 2], 
                   c='black', marker='x', s=60, label='Errors')
    
    ax.set_title(f"{name}\nTest Acc: {results[name]*100:.2f}%")
    if i == 0: ax.legend()

plt.tight_layout()
plt.show()