import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import time

## data
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10)


## NN variables
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

## NN super parameters
lr = 0.1
epoch_num = 500
train_loss_result = []
test_acc = []


## train && test
############################################################
## sgd-momentun
beta = 0.9
m_w, m_b = 0, 0

## adagrad
# v_w, v_b = 0, 0

## RMSProp
# v_w, v_b = 0, 0
# beta = 0.9

## Adam
# beta1, beta2 = 0.9, 0.999
# m_w, m_b = 0, 0
# v_w, v_b = 0, 0
# global_step = 0
############################################################
start_time = time.time()
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
        # updata parameters
        grads = Tape.gradient(loss, [w1, b1])

        def optimizer_SGD(): ## sgd
            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])
            return w1, b1

        def optimizer_SGDM(m_w, m_b): ## sgd-momentun, 一阶动量
            m_w = beta * m_w + (1 - beta) * grads[0]
            m_b = beta * m_b + (1 - beta) * grads[1]
            w1.assign_sub(lr * m_w)
            b1.assign_sub(lr * m_b)
            return w1, b1, m_w, m_b

        def optimizer_Adagrad(v_w, v_b): # adagrad, 二阶动量
            v_w += tf.square(grads[0])
            v_b += tf.square(grads[1])
            w1.assign_sub(lr * grads[0] / tf.sqrt(v_w))
            b1.assign_sub(lr * grads[1] / tf.sqrt(v_b))
            return w1, b1, v_w, v_b

        def optimizer_RMSProp(v_w, v_b): # RMSProp, 二阶动量
            v_w = beta * v_w + (1 - beta) * tf.square(grads[0])
            v_b = beta * v_b + (1 - beta) * tf.square(grads[1])
            w1.assign_sub(lr * grads[0] / tf.sqrt(v_w))
            b1.assign_sub(lr * grads[1] / tf.sqrt(v_b))
            return w1, b1, v_w, v_b

        def optimizer_Adam(m_w, m_b, v_w, v_b, global_step): # Adam, 一阶动量+二阶动量
            global_step += 1
            m_w = beta1 * m_w + (1 - beta1) * grads[0]
            m_b = beta1 * m_b + (1 - beta1) * grads[1]
            v_w = beta2 * v_w + (1 - beta2) * tf.square(grads[0])
            v_b = beta2 * v_b + (1 - beta2) * tf.square(grads[1])

            m_w_corect = m_w / (1 - tf.pow(beta1, int(global_step)))
            m_b_corect = m_b / (1 - tf.pow(beta1, int(global_step)))
            v_w_corect = v_w / (1 - tf.pow(beta2, int(global_step)))
            v_b_corect = v_b / (1 - tf.pow(beta2, int(global_step)))
            w1.assign_sub(lr * m_w_corect / tf.sqrt(v_w_corect))
            b1.assign_sub(lr * m_b_corect / tf.sqrt(v_b_corect))
            return w1, b1, m_w, m_b, v_w, v_b, global_step


        ## call optimizer
        # w1, b1 = optimizer_SGD()
        w1, b1, m_w, m_b = optimizer_SGDM(m_w, m_b)
        # w1, b1, v_w, v_b = optimizer_Adagrad(v_w, v_b)
        # w1, b1, v_w, v_b = optimizer_RMSProp(v_w, v_b) # lr = 0.01
        # w1, b1, m_w, m_b, v_w, v_b, global_step = optimizer_Adam(m_w, m_b, v_w, v_b, global_step)



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
print("Total time is ", time.time()-start_time)


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