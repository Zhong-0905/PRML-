import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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

# AdaBoost + Decision Trees 模型训练

# max_depth 设为 3，增强基分类器的表达能力
base_tree = DecisionTreeClassifier(max_depth=3, random_state=42)

ada_model = AdaBoostClassifier(
    estimator=base_tree, 
    n_estimators=200,    # 迭代次数 200（100不行）
    learning_rate=0.5,   # 降低学习率使收敛更平稳
    random_state=42
)

ada_model.fit(X_train, y_train)

# 性能评估与输出
y_train_pred = ada_model.predict(X_train)
y_test_pred = ada_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"训练集准确率 (Training Accuracy): {train_acc * 100:.2f}%")
print(f"测试集准确率 (Testing Accuracy): {test_acc * 100:.2f}%")

# 可视化分类结果
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制测试集真实标签
ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], X_test[y_test==0, 2], 
           c='skyblue', label='Class 0 (True)', alpha=0.6)
ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], X_test[y_test==1, 2], 
           c='salmon', label='Class 1 (True)', alpha=0.6)

# 标记分类错误的点
errors = (y_test != y_test_pred)
ax.scatter(X_test[errors, 0], X_test[errors, 1], X_test[errors, 2], 
           c='black', marker='x', s=50, label='Misclassified')

ax.set_title(f"Adjusted AdaBoost Classification")
ax.legend()
plt.show()