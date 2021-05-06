import tensorflow as tf
from sklearn import datasets
import numpy as np

x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)
# print(y_data)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
    ]
)

# 数值与one-hot
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy']
              )
model.fit(x_data, y_data, batch_size=32, epochs=500, validation_split=0.2, validation_freq=10)
model.summary()
