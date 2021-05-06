import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Sequential

class BasicBlock(Model):
    def __init__(self, filters, strides, dim_align=False):
        super(BasicBlock, self).__init__()

        self.filters = filters
        self.strides = strides
        self.dim_align = dim_align

        self.c1 = layers.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.Activation('relu')

        self.c2 = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.b2 = layers.BatchNormalization()

        if dim_align:
            self.c_align = layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False)
            self.b_align = layers.BatchNormalization()

        self.a2 = layers.Activation('relu')

    def call(self, inputs):
        residual = inputs
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        x = self.b2(x)
        if self.dim_align:
            residual = self.c_align(residual)
            residual = self.b_align(residual)

        y = self.a2(x + residual) #F(x)+x或F(x)+Wx
        return y



class ResNet18(Model):
    def __init__(self, block_list=[2,2,2,2], init_filters=64, num_class=10):
        super(ResNet18, self).__init__()
        self.block_list = block_list
        self.out_filter = init_filters
        self.num_class = num_class

        self.c1 = layers.Conv2D(init_filters, kernel_size=3, strides=1, padding='same',
                                use_bias=False, kernel_initializer='he_normal')
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.Activation('relu')

        self.blocks = Sequential()

        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                if block_id !=0 and layer_id==0:
                    block = BasicBlock(self.out_filter, strides=2, dim_align=True)
                else:
                    block = BasicBlock(self.out_filter, strides=1, dim_align=False)
                self.blocks.add(block)
            self.out_filter *= 2
        self.p1 = layers.GlobalAveragePooling2D()
        self.f1 = layers.Dense(self.num_class, activation='softmax')


    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)

        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


if __name__ == '__main__':
    ResNet18([2,2,2,2], init_filters=64, num_class=10)