import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


### data ###
input_data = 'abcde'
w_2_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
id_2_onehot = {0: [1., 0., 0., 0., 0.],
               1: [0., 1., 0., 0., 0.],
               2: [0., 0., 1., 0., 0.],
               3: [0., 0., 0., 1., 0.],
               4: [0., 0., 0., 0., 1.]}

x_train = [
    [id_2_onehot[w_2_id['a']], id_2_onehot[w_2_id['b']], id_2_onehot[w_2_id['c']], id_2_onehot[w_2_id['d']]],
    [id_2_onehot[w_2_id['b']], id_2_onehot[w_2_id['c']], id_2_onehot[w_2_id['d']], id_2_onehot[w_2_id['e']]],
    [id_2_onehot[w_2_id['c']], id_2_onehot[w_2_id['d']], id_2_onehot[w_2_id['e']], id_2_onehot[w_2_id['a']]],
    [id_2_onehot[w_2_id['d']], id_2_onehot[w_2_id['e']], id_2_onehot[w_2_id['a']], id_2_onehot[w_2_id['b']]],
    [id_2_onehot[w_2_id['e']], id_2_onehot[w_2_id['a']], id_2_onehot[w_2_id['b']], id_2_onehot[w_2_id['c']]]
]

y_train = [w_2_id['e'],
           w_2_id['a'],
           w_2_id['b'],
           w_2_id['c'],
           w_2_id['d']]

x_train = np.reshape(x_train, (len(x_train), 4,  5))
y_train = np.array(y_train)

np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

# print(x_train)
# print(y_train)

### model ###
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(5, activation='softmax')

])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['sparse_categorical_accuracy'])

ckpt_save_path = './checkpoint/rnn_onehot_4pre1.ckpt'

if os.path.exists(ckpt_save_path+'.index'):
    print('---------load the model-------------')
    model.load_weights(ckpt_save_path)


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_save_path,
                                                 monitor='loss',
                                                 save_best_only=True,
                                                 save_weights_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])
model.summary()


### save weights in txt ###
f = open('weights.txt', 'w')
for v in model.trainable_variables:
    f.write(str(v.name) + '\n')
    f.write(str(v.shape) + '\n')
    f.write(str(v.numpy()) + '\n')

f.close()


### acc && loss ###
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(loss, label='Training loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc, label='Training accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.show()


### pred ###
preNum = int(input('input the num of test alphabet (abcde):'))
for i in range(preNum):
    alphabets_in = input('input 4 alphabets:')
    alphabets = [id_2_onehot[w_2_id[item]] for item in alphabets_in]
    alphabets = np.reshape(alphabets, (1, 4, 5))
    result = model.predict([alphabets])
    pred = tf.argmax(result, axis=1)
    pred = input_data[int(pred)]
    tf.print(alphabets_in + '->' + pred)

