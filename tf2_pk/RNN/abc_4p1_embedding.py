import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

### data ###
input_data = 'abcdefghijklmnopqrstuvwxyz'
w_2_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4,
          'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
          'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14,
          'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19,
          'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}

train_set_scaled = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

x_train, y_train = [], []
for i in range(4, 25):
    x_train.append(train_set_scaled[i-4:i])
    y_train.append(train_set_scaled[i])

x_train = np.reshape(x_train, (len(x_train), 4))
y_train = np.array(y_train)

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

### model ###
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(26, 2),
    tf.keras.layers.SimpleRNN(10),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

ckpt_save_path = './checkpoint/rnn_embedding_4pre1.ckpt'
if os.path.exists(ckpt_save_path + '.index'):
    print('----------- load the model ----------')
    model.load_weights(ckpt_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_save_path,
                                                 monitor='loss',
                                                 save_best_only=True,
                                                 save_weights_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])
model.summary()


### save weights to txt ###
f = open('./weights.txt', 'w')
for v in model.trainable_variables:
    f.write(str(v.name) + '\n')
    f.write(str(v.shape) + '\n')
    f.write(str(v.numpy()) + '\n')
f.close()

### acc && loss ###
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(loss, label='Training Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc, label='Training Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.show()

### pred ###
preNum = int(input('input the number of test alphabet:'))
for i in range(preNum):
    alphabet_in = input('input 4 alphabet:')
    alphabet = [w_2_id[item] for item in alphabet_in]
    alphabet = np.reshape(alphabet, (1, 4))
    result = model.predict(alphabet)
    pred = tf.argmax(result, axis=1)
    pred = input_data[int(pred)]
    tf.print(alphabet_in + '->' + pred)
