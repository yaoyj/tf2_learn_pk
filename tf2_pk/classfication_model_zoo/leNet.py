from tensorflow.keras.models import Model
from tensorflow.keras import layers

"""LeNet  AlexNet  VGG  Inception  ResNet"""


## 5 layers: 2 convs + 3 fc
class LeNet(Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = layers.Conv2D(filters=6, kernel_size=(5, 5), activation='sigmoid')
        self.p1 = layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.c2 = layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid')
        self.p2 = layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = layers.Flatten()
        self.f1 = layers.Dense(120, activation='sigmoid')
        self.f2 = layers.Dense(84, activation='sigmoid')
        self.f3 = layers.Dense(10, activation='softmax')


    def call(self, x): # x shape: [32, 32, 3]
        x = self.c1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y


if __name__=='__main__':
    model = LeNet()
