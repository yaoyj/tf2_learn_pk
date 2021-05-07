import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt



device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
else:
    print('Found GPU at {}'.format(device_name))


### data ###
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
print(x_train.shape, x_test.shape)
print(x_train.dtype)

# '''
### model ###
class ResnetBlock(Model):
    def __init__(self, filters, strides=1, dim_align=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.dim_align = dim_align

        self.c1 = tf.keras.layers.Conv2D(filters, (3, 3), strides=strides, padding='same',
                                         use_bias=False, input_shape=(32, 32, 3))
        self.b1 = tf.keras.layers.BatchNormalization()
        self.a1 = tf.keras.layers.Activation('relu')

        self.c2 = tf.keras.layers.Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = tf.keras.layers.BatchNormalization()

        if dim_align:
            self.c_align = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.b_align = tf.keras.layers.BatchNormalization()

        self.a2 = tf.keras.layers.Activation('relu')

    @tf.function
    def call(self, inputs):
        residual = inputs
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        x = self.b2(x)

        if self.dim_align:
            residual = self.c_align(inputs)
            residual = self.b_align(residual)

        y = self.a2(x + residual)
        return y

class ResNet18(Model):
    def __init__(self, block_list, initial_filters=64):# block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)
        self.block_list = block_list
        self.out_filters = initial_filters

        self.c1 = tf.keras.layers.Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = tf.keras.layers.BatchNormalization()
        self.a1 = tf.keras.layers.Activation('relu')

        self.blocks = tf.keras.Sequential()
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                if block_id != 0 and layer_id == 0:
                    block = ResnetBlock(filters=self.out_filters, strides=2, dim_align=True)
                else:
                    block = ResnetBlock(filters=self.out_filters, strides=1, dim_align=False)
                self.blocks.add(block)
            self.out_filters *= 2
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    @tf.function
    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.blocks(x)

        x = self.p1(x)
        y = self.f1(x)
        return y

model = ResNet18([2, 2, 2, 2])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# '''
#### comment start for evaluate ####
model_filename = './results/cifar_resnet18.h5'
model.build(input_shape=(None, 32, 32, 3))
# model.summary()

try:
    model.load_weights(model_filename)
    print('---------- load the model ----------')
except:
    print('--------- failed load model -----------')


### train ###
batch_size = 256
epochs = 10
patience = 30
logdir = './results/tb_results'

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('learning rate:', lr)
    return lr

callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir),
             tf.keras.callbacks.ModelCheckpoint(filepath=model_filename,
                                                save_weights_only=True,
                                                save_best_only=True,
                                                verbose=1),
             tf.keras.callbacks.EarlyStopping(patience=patience),
             tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                  cooldown=0,
                                                  patience=6,
                                                  min_lr=1e-5),
             # tf.keras.callbacks.LearningRateScheduler(lr_schedule)
             ]

train_imagegen = ImageDataGenerator(rotation_range=20,
                                    # rescale=1./255,
                                    width_shift_range=.15,
                                    height_shift_range=.15,
                                    zoom_range=.15,
                                    horizontal_flip=True)

train_imagegen.fit(x_train)
history = model.fit_generator(train_imagegen.flow(x_train, y_train, batch_size=batch_size),
                              steps_per_epoch=(len(x_train))/batch_size,
                              epochs=epochs,
                              validation_data=(x_test, y_test),
                              workers=4,
                              callbacks=callbacks)

model.summary()


### plot acc && loss ###
def plot_train_history(history, train_metric, val_metric):
    plt.plot(history.history[train_metric])
    plt.plot(history.history[val_metric])
    plt.xlabel('epochs')
    plt.ylabel(train_metric)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


plot_train_history(history, 'loss', 'val_loss')
plot_train_history(history, 'sparse_categorical_accuracy', 'val_sparse_categorical_accuracy')
#### comment end for evaluate ####
# '''

### model evaluate && predict (.h5) ####
label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
              5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}


model_filename = './results/cifar_resnet18.h5'
try:
    model.load_weights(model_filename)
    print('---------- load the model ----------')
except:
    print('--------- failed load model -----------')


model.evaluate(x_test, y_test)
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
print(pred[:10])

def plot_prediction(images, labels, predictions, index, nums=5):
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    if nums > 10:
        nums = 10
    for i in range(nums):
        ax = plt.subplot(2, 5, i+1)
        ax.imshow(images[index])
        # title = str(i) + label_dict[labels[index][0]]
        title = label_dict[labels[index][0]]
        if len(predictions) > 0:
            title += '==>' + label_dict[predictions[index]]
        ax.set_title(title, fontsize=12)
        index += 1
    plt.show()

plot_prediction(x_test, y_test, pred, 0, 10)


### saved_model (.pb)  ###
tf.saved_model.save(model, "./results/saved/1")
model = tf.saved_model.load("./results/saved/1")
print(x_test.shape)
pred = model(tf.cast(x_test, tf.float32))
pred = np.argmax(pred, axis=1)
print(pred[:10])
