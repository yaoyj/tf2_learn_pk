import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# show predictions on img[i] and its pred label score.
def plot_image(id, predictions_array, true_label, imgs):
    true_label, img = true_label[id], imgs[id]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    pred_label = np.argmax(predictions_array)
    if pred_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[pred_label], 100*np.max(predictions_array),
                                         class_names[true_label]), color=color)

def plot_value_array(id, predictions_array, true_label):
    true_label = true_label[id]
    plt.grid(False)
    plt.xticks(range(10))
    # plt.xticks(range(10), class_names, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    pred_label = np.argmax(predictions_array)

    thisplot[pred_label].set_color('red')
    thisplot[true_label].set_color('blue')



def call_plot_result_one_img(id, predictions_array, true_label, imgs):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(id, predictions_array[id], true_label, imgs)
    plt.subplot(1, 2, 2)
    plot_value_array(id, predictions_array[id], true_label)
    plt.show()


# # Plot multi images pred.
def call_plot_result_multi_img(predictions_array, true_label, imgs):
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions_array[i], true_label, imgs)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions_array[i], true_label)
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # print(train_images.shape)
    # print(train_labels.shape)
    # print(train_labels)
    # print(train_images[0])

    train_images, test_images = train_images/255.0, test_images/255.0
    # print(train_images[0])
    # plt.figure()
    # plt.set_cmap('gray')
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.show()

    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5, 5, i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary) #白底
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()


    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    # model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    #
    # model.fit(train_images, train_labels, epochs=10)
    # test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    # print('\nTest accuracy:', test_acc)

    model.fit(train_images, train_labels, batch_size=32, epochs=5,
              validation_data=(test_images, test_labels), validation_freq=10)
    #
    #
    #
    # ### prediction
    # pred_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # predictions = pred_model.predict(test_images)
    # pred_class_num = np.argmax(predictions, axis=1)
    # # call_plot_result_one_img(0, predictions, test_labels, test_images)
    # call_plot_result_multi_img(predictions, test_labels, test_images)

    ### pred one real image
    test_img = test_images[1]
    test_img = np.expand_dims(test_img, axis=0)
    pred_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = pred_model.predict(test_img)
    pred_class_num = np.argmax(predictions, axis=1)
    pred_class_name = class_names[pred_class_num[0]]
    print('The image class is {}'.format(pred_class_name))
    plot_value_array(1, predictions[0], test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.show()



