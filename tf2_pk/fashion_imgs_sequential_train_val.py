import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import pathlib
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


"""customer data-->data augment-->save ckpt-->continue train from ckpt-->results vis-->get model paras"""

def generateDataset(img_dir, txt_path):
    f = open(txt_path, 'r')
    lines = f.readlines()
    f.close()
    x, y = [], []
    for cont in lines:
        item = cont.split()
        image = Image.open(pathlib.Path(img_dir, item[0]))
        image = np.array(image.convert('L')) #图片变为8位宽灰度值的numpy
        image = image/255.0  # normlize
        x.append(image)
        y.append(item[1])
        print('Loading: ' + cont)
    x = np.array(x)
    y = np.array(y)
    y = y.astype(np.int64)
    return x, y


def show_mnist_images(data, num, batch_data=False):
    fig = plt.figure(figsize=(20, 2))
    plt.axis('off')
    plt.set_cmap('gray')
    if not batch_data:
        for i in range(0, num):
            ax = fig.add_subplot(1, num, i + 1)
            ax.imshow(data[i])
        fig.suptitle('Subset of Original Training Images', fontsize=20)
        plt.savefig('org_data.jpg')
        plt.show()


    else: ## augmented imgs
        for x_batch in img_gen_train.flow(data, batch_size=num, shuffle=False):
            for i in range(0, num):
                ax = fig.add_subplot(1, num, i + 1)
                ax.imshow(np.squeeze(x_batch[i]))
            fig.suptitle('Augmented Images', fontsize=20)
            plt.savefig('augmented_data.jpg')
            plt.show()
            break

def save_weights_value(path, model):
    f = open(path, 'w')
    # print(model.trainable_variables)
    for w in model.trainable_variables:
        f.write(str(w.name) + '\n')
        f.write(str(w.shape) + '\n')
        f.write(str(w.numpy()) + '\n')
    f.close()

if __name__=='__main__':
    ## data
    mnist_root = '/media/windows/e/Public_Datasets/fashion_mnist/fashion_image_label'
    dataset_type = ['fashion_train_jpg_60000', 'fashion_test_jpg_10000']
    txt_train = pathlib.Path(mnist_root, dataset_type[0]+'.txt')
    txt_test = pathlib.Path(mnist_root, dataset_type[1]+'.txt')
    img_dir_train = pathlib.Path(mnist_root, dataset_type[0])
    img_dir_test = pathlib.Path(mnist_root, dataset_type[1])
    npy_datasets = ['x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy']
    npy_save_dir = pathlib.Path(mnist_root, 'np_data')


    if npy_save_dir.exists():
        print('----------Load datasets----------')
        x_train_ = np.load(pathlib.Path(npy_save_dir, npy_datasets[0]))
        y_train = np.load(pathlib.Path(npy_save_dir, npy_datasets[1]))
        x_test_ = np.load(pathlib.Path(npy_save_dir, npy_datasets[2]))
        y_test = np.load(pathlib.Path(npy_save_dir, npy_datasets[3]))
        x_train = np.reshape(x_train_, (len(x_train_), 28, 28))
        x_test = np.reshape(x_test_, (len(x_test_), 28, 28))
        print('org test img shape:', x_test.shape)
        print('org test label shape:', y_test.shape)

    else:
        print('----------Generate datasets----------')
        x_train, y_train = generateDataset(img_dir_train, txt_train)
        x_test, y_test = generateDataset(img_dir_test, txt_test)
        print('----------Save datasets----------')
        x_train_ = np.reshape(x_train, (len(x_train), -1))
        x_test_ = np.reshape(x_test, (len(x_test), -1))
        pathlib.Path(npy_save_dir).mkdir(parents=True, exist_ok=True)

        np.save(pathlib.Path(npy_save_dir, npy_datasets[0]), x_train_)
        np.save(pathlib.Path(npy_save_dir, npy_datasets[1]), y_train)
        np.save(pathlib.Path(npy_save_dir, npy_datasets[2]), x_test_)
        np.save(pathlib.Path(npy_save_dir, npy_datasets[3]), y_test)


    ## data augment
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)) # 后面增加1维
    img_gen_train = ImageDataGenerator(rescale=1./1.,
                                       rotation_range=45,
                                       width_shift_range=0.15,
                                       height_shift_range=0.15,
                                       horizontal_flip=True,
                                       zoom_range=0.5
                                       )

    img_gen_train.fit(x_train)

    print('augmented train img shape', x_train.shape)
    ## vis augmeted data
    # num_show = 15
    # show_batch_data = x_train[:num_show]
    # show_org_data = np.squeeze(show_batch_data)
    # show_mnist_images(show_org_data, num_show, batch_data=False)
    # show_mnist_images(show_batch_data, num_show, batch_data=True)



    ## model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ]
    )

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['sparse_categorical_accuracy'])

    ## save_model
    ckpt_save_dir = pathlib.Path('./checkpoint')
    if not ckpt_save_dir.exists():
        pathlib.Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True)
    ckpt_save_path = pathlib.Path(ckpt_save_dir, 'mnist.ckpt')
    if pathlib.Path(str(ckpt_save_path) + '.index').is_file():
        print('----------load model----------')
        model.load_weights(ckpt_save_path)

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_save_path,
                                                       save_best_only=True,
                                                       save_weights_only=True)
    # model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
    train_op = model.fit(img_gen_train.flow(x_train, y_train, batch_size=32), epochs=5,
                         validation_data=(x_test, y_test), validation_freq=1, callbacks=[ckpt_callback])
    model.summary()

    ## print parametes
    print('----------print model parameters----------')
    f_path = './weights.txt'
    save_weights_value(f_path, model)
    # print(model.trainable_variables)

    ## show loss && acc
    acc_train = train_op.history['sparse_categorical_accuracy']
    acc_val = train_op.history['val_sparse_categorical_accuracy']
    loss_train = train_op.history['loss']
    loss_val = train_op.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc_train, label='Training Accuracy')
    plt.plot(acc_val, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_train, label='Training Loss')
    plt.plot(loss_val, label='Validation Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()

