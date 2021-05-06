import tensorflow as tf
from sklearn import datasets
import pandas
from pandas import DataFrame
from matplotlib import pyplot as plt
import numpy as np


## data
x_data = datasets.load_iris().data  # 150*4
y_data = datasets.load_iris().target # 150
data_num = x_data.shape[0]
print(data_num)


## vis data table
data_table = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
pandas.set_option('display.unicode.east_asian_width', True) # 设置列名对齐
data_table['类别'] = y_data

# tf.random.set_seed(116)
train_size = int(0.8*data_num)
x_data = tf.cast(x_data, tf.float32)
datasets = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(buffer_size=data_num)
train_data = datasets.take(train_size).batch(32)
test_data = datasets.skip(train_size).batch(10)
# test_data = test_data.as_numpy_iterator()

## model
# NN variables
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

def iris_nn(input_data):
    y = tf.matmul(input_data, w1) + b1
    y = tf.nn.softmax(y)
    return y


## NN super parameters
lr = 0.12
epoch_num = 500
train_loss_result = []
test_acc = []

## train && test
for epoch in range(epoch_num):
    # train
    loss_all = 0
    for step, (x_train, y_train) in enumerate(train_data): # steps_num=4 (data_num/batch)
        with tf.GradientTape() as Tape:
            y = iris_nn(x_train)
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()
        grads = Tape.gradient(loss, [w1, b1])
        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr*grads[1])
    avg_epoch_loss = loss_all/(int(step+1))
    print("Epoch {}, loss: {}".format(epoch, avg_epoch_loss))
    train_loss_result.append(avg_epoch_loss)

    # test
    total_correct_num, total_test_num = 0, 0
    for x_test, y_test in test_data:
        y = iris_nn(x_test)
        pred = tf.argmax(y, axis=1) # pred label
        pred = tf.cast(pred, y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32) #bool-->binary
        correct_num = tf.reduce_sum(correct)
        total_correct_num += int(correct_num)
        total_test_num += y_test.shape[0]

    acc = total_correct_num/total_test_num
    test_acc.append(acc)
    print("Test acc:", acc)


## plot loss
plt.title("Loss Function Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(train_loss_result, label="$Loss$")# 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()
plt.show()


## plot acc
plt.title("Acc Curve")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()
