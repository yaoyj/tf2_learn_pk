import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

## plot acc and loss
def plot_acc_loss(acc, acc_val, loss, loss_val, epochs):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), acc, label='Traing Accuracy')
    plt.plot(range(epochs), acc_val, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), loss, label='Traing Loss')
    plt.plot(range(epochs), loss_val, label='Validation Loss')
    plt.title('Loss Curves')
    plt.legend(loc='lower right')
    plt.show()


## save weights to txt
def save_weights_to_txt(model):
    f = open('weights.txt', 'w')
    for params in model.trainable_variables:
        f.write(str(params.name) + '\n')
        f.write(str(params.shape) + '\n')
        f.write(str(params.numpy()) + '\n')
    f.close()

## data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
class_name_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# print(x_train.shape)
# print(x_train[0])
# print(y_train[:20])
# plt.imshow(x_train[19])
# plt.show()

## base conv model
class BaseConv(Model):
    def __init__(self):
        super(BaseConv, self).__init__()
        self.c1 = layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same')
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.Activation('relu')
        self.p1 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = layers.Dropout(rate=0.2)

        self.flatten = layers.Flatten()
        self.f1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dropout(rate=0.2)
        self.f2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y



## model instance
from classfication_model_zoo import leNet, alexNet, vgg, inception, resNet

# model = BaseConv()
model = leNet.LeNet()
# model = alexNet.AlexNet32()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])


## train
import os
ckpt_save_path = './checkpoint/BaseConv.ckpt'
if os.path.exists(ckpt_save_path + '.index'):
    print('loading saved checkpoint model')
    model.load_weights(ckpt_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
history = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), callbacks=[cp_callback])
model.summary()

## save weights to txt
save_weights_to_txt(model)


## plot acc and loss
# acc_train = history.history['sparse_categorical_accuracy']
# acc_val = history.history['val_sparse_categorical_accuracy']
# loss_train = history.history['loss']
# loss_val = history.history['val_loss']
# plot_acc_loss(acc_train, acc_val, loss_train, loss_val, epochs=10)
