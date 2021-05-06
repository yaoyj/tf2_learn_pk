import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

'''model-->model.load_weights-->model.predict'''

def image_process(image_path, binary=False):
    image = Image.open(image_path)
    image = image.resize((28, 28), Image.ANTIALIAS)
    image = np.array(image.convert('L'))

    if not binary:
        image = 255 - image
    else:
        for i in range(28):
            for j in range(28):
                if image[i][j] < 200:
                    image[i][j] = 255
                else:
                    image[i][j] = 0

    plt.set_cmap('gray')
    plt.imshow(image)
    plt.show()
    image = image / 255.0
    return image


ckpt_save_path = './checkpoint/mnist.ckpt'

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, 'relu'),
    tf.keras.layers.Dense(10, 'softmax')
])

model.load_weights(ckpt_save_path)

if __name__ == '__main__':
    image_dir = Path('/media/windows/e/Public_Datasets/mnist/new_test')
    for img_path in image_dir.glob('*'):
        print(img_path)
        image = image_process(img_path, binary=False)
        image_pred = image[tf.newaxis, ...]
        # print(image_pred.shape)
        pred_result = model.predict(image_pred)
        pred_y = tf.argmax(pred_result, axis=1)
        # print(pred_result)
        tf.print(pred_y)