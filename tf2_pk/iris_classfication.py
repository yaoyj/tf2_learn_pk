import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

# tf.compat.v1.enable_eager_execution()


## data
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# print(x_data)
# print(x_data)
# print(y_data.shape)
# print(y_data.shape)

x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10)
# for i in test_data:
#     print(i)

## NN variables
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

## NN super parameters
lr = 0.1
epoch_num = 500
train_loss_result = []
test_acc = []


## train && test
for epoch in range(epoch_num):
    # train
    loss_all = 0
    for step, (x_train, y_train) in enumerate(train_data): # steps_num=4 (data_num/batch)
        with tf.GradientTape() as Tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
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
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
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