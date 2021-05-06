import tensorflow as tf
import numpy as np
import os
import matplotlib.pylab as plt

### data ###
input_data = "abcde"
w_2_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
id_2_onehot = {0: [1., 0., 0., 0., 0.],
               1: [0., 1., 0., 0., 0.],
               2: [0., 0., 1., 0., 0.],
               3: [0., 0., 0., 1., 0.],
               4: [0., 0., 0., 0., 1.]}

x_train = [id_2_onehot[w_2_id['a']],
           id_2_onehot[w_2_id['b']],
           id_2_onehot[w_2_id['c']],
           id_2_onehot[w_2_id['d']],
           id_2_onehot[w_2_id['e']]]

y_train = [w_2_id['b'],
           w_2_id['c'],
           w_2_id['d'],
           w_2_id['e'],
           w_2_id['a']]

print(x_train)
print(y_train)

# for i in range(5):
#     onehot_code = tf.one_hot(i, depth=5)
#     print(onehot_code)

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train = np.reshape(x_train, (len(x_train), 1, 5)) # RNN input shape
y_train = np.array(y_train)
print(x_train)
print(y_train)


### model ###
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

ckpt_save_path = './checkpoint/rnn_onehot_1p1.ckpt'
if os.path.exists(ckpt_save_path + '.index'):
    print('------------load the model--------------')
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


### loss curve ###
loss = history.history['loss']
acc = history.history['sparse_categorical_accuracy']

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
preNum = int(input('input the number of test alphabet (a:e):'))
for i in range(preNum):
    alphabet_in = input('input test alphabet:')
    alphabet = [id_2_onehot[w_2_id[alphabet_in]]]
    alphabet = np.reshape(alphabet, (1, 1, 5))
    result = model.predict([alphabet])
    pred = tf.argmax(result, axis=1)
    pred = int(pred)
    tf.print(alphabet_in + '->' + input_data[pred])
