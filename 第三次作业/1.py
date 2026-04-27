import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 数据加载
dataset = pd.read_csv('LSTM-Multivariate_pollution.csv', parse_dates=['date'], index_col=0)
dataset.index.name = 'date'

# 预处理
values = dataset.values
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4]) # wnd_dir编码成数字
values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# 使用前1小时预测当前
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)

# 划分数据集（训练、测试）
values = reframed.values
n_train_hours = 365 * 24 * 4
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# 模型
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# ing
history = model.fit(train_X, train_y, epochs=50, batch_size=72, 
                    validation_data=(test_X, test_y), verbose=1, shuffle=False)

# MAE 损失曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train MAE (Loss)')
plt.plot(history.history['val_loss'], label='Test MAE (Loss)')
plt.title('Model Training Loss (MAE)')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 执行预测与反归一化
yhat = model.predict(test_X)
test_X_2d = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# 反归一化预测值
inv_yhat = np.concatenate((yhat, test_X_2d[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# 反归一化真实值
test_y_reshaped = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y_reshaped, test_X_2d[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print(f'\nFinal Test RMSE: {rmse:.3f}')
# 结果可视化
plt.figure(figsize=(12, 6))
plt.plot(inv_y[100:300], label='Actual Pollution', color='blue', alpha=0.7)
plt.plot(inv_yhat[100:300], label='Predicted Pollution', color='red', linestyle='--')
plt.title('PM2.5 Prediction Comparison (Snapshot)')
plt.xlabel('Time (Hours)')
plt.ylabel('Concentration')
plt.legend()
plt.show()