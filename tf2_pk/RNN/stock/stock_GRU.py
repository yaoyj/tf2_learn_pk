import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os

np.set_printoptions(threshold=np.inf)

### data ###
stock_data = pd.read_csv('./SH600519.csv')
num_rows, num_cols = stock_data.shape

train_data = stock_data.iloc[:num_rows-300, 2:3].values
test_data = stock_data.iloc[num_rows-300:, 2:3].values

# norm
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.fit_transform(test_data)

# split label
x_train, y_train, x_test, y_test = [], [], [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))


### model ###
model = tf.keras.Sequential([
    tf.keras.layers.GRU(80, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.MeanSquaredError()) # only loss

ckpt_save_path = './checkpoint/stock_gru.ckpt'
if os.path.exists(ckpt_save_path + '.index'):
    print('----------- load the model --------------')
    model.load_weights(ckpt_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_save_path,
                                                 save_best_only=True,
                                                 save_weights_only=True)
history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test),
                    callbacks=[cp_callback])

model.summary()


### save weights to txt ###
f = open('./weights.txt', 'w')
for v in model.trainable_variables:
    f.write(str(v.name) + '\n')
    f.write(str(v.shape) + '\n')
    f.write(str(v.numpy()) + '\n')
f.close()


### loss ##
loss_train = history.history['loss']
loss_test = history.history['val_loss']
plt.plot(loss_train, color='r', label='train loss')
plt.plot(loss_test, color='b', label='test loss')
plt.title('Training and Testing loss curves')
plt.legend()
plt.show()


### pred ###
pred_stock_price = model.predict(x_test)
pred_stock_price = scaler.inverse_transform(pred_stock_price)
y_test = np.reshape(y_test, (y_test.shape[0], 1))
real_stock_price = scaler.inverse_transform(y_test)

plt.plot(real_stock_price, color='r', label='real price of MaoTai')
plt.plot(pred_stock_price, color='b', label='pred price of MaoTai')
plt.title('MaoTai Stock Price Prediction')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()

mse = mean_squared_error(real_stock_price, pred_stock_price)
rmse = math.sqrt(mse)
mae = mean_absolute_error(real_stock_price, pred_stock_price)
print('均方误差： %.6f' % mse)
print('均方根误差： %.6f' % rmse)
print('平均绝对误差： %.6f' % mae)