from tensorflow.keras.models import Model
from tensorflow.keras import layers

# input_shape [224, 224, 3]
# 13 conv + 3 fc
class VGG16(Model):
    def __init__(self, num_class=1000):
        super(VGG16, self).__init__()
        self.c1 = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.Activation('relu')
        self.c2 = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')
        self.b2 = layers.BatchNormalization()
        self.a2 = layers.Activation('relu')
        self.p1 = layers.MaxPool2D(pool_size=(2,2), strides=2)

        self.c3 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b3 = layers.BatchNormalization()
        self.a3 = layers.Activation('relu')
        self.c4 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b4 = layers.BatchNormalization()
        self.a4 = layers.Activation('relu')
        self.p2 = layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.c5 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b5 = layers.BatchNormalization()
        self.a5 = layers.Activation('relu')
        self.c6 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b6 = layers.BatchNormalization()
        self.a6 = layers.Activation('relu')
        self.c7 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b7 = layers.BatchNormalization()
        self.a7 = layers.Activation('relu')
        self.p3 = layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.c8 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b8 = layers.BatchNormalization()
        self.a8 = layers.Activation('relu')
        self.c9 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b9 = layers.BatchNormalization()
        self.a9 = layers.Activation('relu')
        self.c10 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b10 = layers.BatchNormalization()
        self.a10 = layers.Activation('relu')
        self.p4 = layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.c11 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b11 = layers.BatchNormalization()
        self.a11 = layers.Activation('relu')
        self.c12 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b12 = layers.BatchNormalization()
        self.a12 = layers.Activation('relu')
        self.c13 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b13 = layers.BatchNormalization()
        self.a13 = layers.Activation('relu')
        self.p5 = layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = layers.Flatten()
        self.f1 = layers.Dense(4096, activation='relu')
        self.d1 = layers.Dropout(rate=0.5)
        self.f2 = layers.Dense(4096, activation='relu')
        self.d2 = layers.Dropout(rate=0.5)
        self.f3 = layers.Dense(num_class, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p1(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p2(x)

        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        x = self.p3(x)

        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        x = self.p4(x)

        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        x = self.p5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y

# 16 conv + 3 fc
class VGG19(Model):
    def __init__(self, num_class=1000):
        super(VGG16, self).__init__()
        self.c1 = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.Activation('relu')
        self.c2 = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same')
        self.b2 = layers.BatchNormalization()
        self.a2 = layers.Activation('relu')
        self.p1 = layers.MaxPool2D(pool_size=(2,2), strides=2)

        self.c3 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b3 = layers.BatchNormalization()
        self.a3 = layers.Activation('relu')
        self.c4 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b4 = layers.BatchNormalization()
        self.a4 = layers.Activation('relu')
        self.p2 = layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.c5 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b5 = layers.BatchNormalization()
        self.a5 = layers.Activation('relu')
        self.c6 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b6 = layers.BatchNormalization()
        self.a6 = layers.Activation('relu')
        self.c7 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b7 = layers.BatchNormalization()
        self.a7 = layers.Activation('relu')
        self.c8 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b8 = layers.BatchNormalization()
        self.a8 = layers.Activation('relu')
        self.p3 = layers.MaxPool2D(pool_size=(2, 2), strides=2)


        self.c9 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b9 = layers.BatchNormalization()
        self.a9 = layers.Activation('relu')
        self.c10 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b10 = layers.BatchNormalization()
        self.a10 = layers.Activation('relu')
        self.c11 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b11 = layers.BatchNormalization()
        self.a11 = layers.Activation('relu')
        self.c12 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b12 = layers.BatchNormalization()
        self.a12 = layers.Activation('relu')
        self.p4 = layers.MaxPool2D(pool_size=(2, 2), strides=2)


        self.c13 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b13 = layers.BatchNormalization()
        self.a13 = layers.Activation('relu')
        self.c14 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b14 = layers.BatchNormalization()
        self.a14 = layers.Activation('relu')
        self.c15 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b15 = layers.BatchNormalization()
        self.a15 = layers.Activation('relu')
        self.c16 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b16 = layers.BatchNormalization()
        self.a16 = layers.Activation('relu')
        self.p5 = layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = layers.Flatten()
        self.f1 = layers.Dense(4096, activation='relu')
        self.d1 = layers.Dropout(rate=0.5)
        self.f2 = layers.Dense(4096, activation='relu')
        self.d2 = layers.Dropout(rate=0.5)
        self.f3 = layers.Dense(num_class, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p1(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p2(x)

        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)
        x = self.p3(x)

        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)
        x = self.p4(x)

        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        x = self.c14(x)
        x = self.b14(x)
        x = self.a14(x)
        x = self.c15(x)
        x = self.b15(x)
        x = self.a15(x)
        x = self.c16(x)
        x = self.b16(x)
        x = self.a16(x)
        x = self.p5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y


