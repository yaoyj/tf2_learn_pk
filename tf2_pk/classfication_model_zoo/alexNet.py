from tensorflow.keras.models import Model
from tensorflow.keras import layers


## 8 layers: 5 convs + 3 fc
# relu, dropout, input shape [224, 224, 3]
class AlexNet(Model):
    def __init__(self, num_class=10):
        super(AlexNet, self).__init__()
        self.c1 = layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4)
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.Activation('relu')
        self.p1 = layers.MaxPool2D(pool_size=(3, 3), strides=2)

        self.c2 = layers.Conv2D(filters=256, kernel_size=(5, 5))
        self.b2 = layers.BatchNormalization()
        self.a2 = layers.Activation('relu')
        self.p2 = layers.MaxPool2D(pool_size=(3, 3), strides=2)

        self.c3 = layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')
        self.c4 = layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')
        self.c5 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        self.p5 = layers.MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = layers.Flatten()
        self.f1 = layers.Dense(2048, activation='relu')
        self.d1 = layers.Dropout(rate=0.5)
        self.f2 = layers.Dense(2048, activation='relu')
        self.d2 = layers.Dropout(rate=0.5)
        self.f3 = layers.Dense(num_class, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.p5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y


# input shape [32, 32, 3]
class AlexNet32(Model):
    def __init__(self, num_class=10):
        super(AlexNet32, self).__init__()
        self.c1 = layers.Conv2D(filters=96, kernel_size=(3, 3))
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.Activation('relu')
        self.p1 = layers.MaxPool2D(pool_size=(3, 3), strides=2)

        self.c2 = layers.Conv2D(filters=256, kernel_size=(3, 3))
        self.b2 = layers.BatchNormalization()
        self.a2 = layers.Activation('relu')
        self.p2 = layers.MaxPool2D(pool_size=(3, 3), strides=2)

        self.c3 = layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')
        self.c4 = layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')
        self.c5 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        self.p5 = layers.MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = layers.Flatten()
        self.f1 = layers.Dense(2048, activation='relu')
        self.d1 = layers.Dropout(rate=0.5)
        self.f2 = layers.Dense(2048, activation='relu')
        self.d2 = layers.Dropout(rate=0.5)
        self.f3 = layers.Dense(num_class, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.p5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y




