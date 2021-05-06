import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Sequential

# c-b-a(relu)
class CommonConv(Model):
    def __init__(self, ch=16, kernel_size=1, strides=1, padding='same'):
        super(CommonConv, self).__init__()
        self.module = Sequential([
            layers.Conv2D(filters=ch, kernel_size=kernel_size, strides=strides, padding=padding),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])
    def call(self, x):
        x = self.module(x)
        return x


class InceptionBlock(Model):
    def __init__(self, channel=16, strides=1):
        super(InceptionBlock, self).__init__()
        self.c1 = CommonConv(ch=channel, strides=strides)
        self.c2_1 = CommonConv(ch=channel, strides=strides)
        self.c2_2 = CommonConv(ch=channel, kernel_size=3, strides=1)
        self.c3_1 = CommonConv(ch=channel, strides=strides)
        self.c3_2 = CommonConv(ch=channel, kernel_size=5, strides=1)
        self.p4_1 = layers.MaxPool2D(pool_size=3, strides=1, padding='same')
        self.c4_2 = CommonConv(ch=channel, strides=strides)

    def call(self, x):
        x1 = self.c1(x)
        x2 = self.c2_1(x)
        x2 = self.c2_2(x2)
        x3 = self.c3_1(x)
        x3 = self.c3_2(x3)
        x4 = self.p4_1(x)
        x4 = self.c4_2(x4)
        y = tf.concat([x1, x2, x3, x4], axis=3) # channel concat
        return y


# 22 layers
class InceptionV1_10(Model):
    def __init__(self, num_blocks, num_class, channel=16):
        super(InceptionV1_10, self).__init__()
        self.channel = channel

        ## layers
        self.c1 = CommonConv(kernel_size=3)
        self.blocks = Sequential()
        for b_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlock(self.channel, strides=2)
                else:
                    block = InceptionBlock(self.channel, strides=1)
                self.blocks.add(block)
            self.channel *= 2
        self.p1 = layers.GlobalAveragePooling2D()
        self.f1 = layers.Dense(num_class, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y




if __name__=='__main__':
    model = InceptionV1_10(num_blocks=2, num_class=10)