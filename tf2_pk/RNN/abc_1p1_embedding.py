import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


### data ###
input_data = 'abcde'
w_2_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}

x_train = [w_2_id['a'], w_2_id['b'], w_2_id['c'], w_2_id['d'], w_2_id['e']]
y_train = [w_2_id['b'], w_2_id['c'], w_2_id['d'], w_2_id['e'], w_2_id['a']]

x_train = np.reshape(x_train, (len(x_train), 1))
y_train = np.array(y_train)

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)


### model ###
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5, 2),
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(5, activation='softmax')
])


model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])
ckpt_save_path = './checkpoint/rnn_embedding_1pre1.ckpt'
if os.path.exists(ckpt_save_path + '.index'):
    print('-------- load the model ------------')
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


### acc && loss ##
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
    alphabet_in = input('input test alphabet:')
    alphabet = w_2_id[alphabet_in]
    alphabet = np.reshape(alphabet, (1, 1))
    result = model.predict(alphabet)
    pred = tf.argmax(result, axis=1)
    pred = input_data[int(pred)]
    tf.print(alphabet_in + '->' + pred)


