import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image
from pylearn2.expr.preprocessing import global_contrast_normalize

import matplotlib.pyplot as plt


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def mnist_imshow(img):
    plt.imshow(img.reshape([28, 28]), cmap="gray")
    plt.axis('off')


def face_imshow(img):
    plt.imshow(img.reshape([112, 92]), cmap="gray")
    plt.axis('off')


def one_hot_preds(preds):
    t = np.argmax(preds, axis=1)
    r = np.zeros(preds.shape)
    for i in range(t.shape[0]):
        r[i, t[i]] = 1
    return r


def one_hot_class(a):
    b = np.zeros((len(a), np.max(a).astype(int) + 1), np.float32)
    b[np.arange(len(a)), a.astype(int)] = 1
    return b


def unpack_facedataset(path='/home/yash/Documents/Attacks/Guillaume-Freisz-project/orl_faces'
                       , sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as err:
                    print("I/O error({0}): {1}".format(err))
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c + 1
    X = (np.array(X).astype(np.float32) / 255).reshape(len(X), 92 * 112)
    y = np.array(y).astype(np.float32)
    X = X.reshape(40, 10, 112 * 92)
    y = y.reshape(40, 10)

    train_x, test_x = X[:, 0:7, :], X[:, 7:10, :]
    train_y, test_y = y[:, 0:7], y[:, 7:10]
    train_x, test_x, train_y, test_y = train_x.reshape(40 * 7, 112 * 92), test_x.reshape(40 * 3,
                                                                                         112 * 92), train_y.reshape(
        40 * 7), test_y.reshape(40 * 3)

    return train_x, test_x, one_hot_class(train_y), one_hot_class(test_y)


def normalize(img, prep, img_shape):
    img = prep.inverse(img.reshape(1, -1))[0]
    img /= np.abs(img).max()
    img = np.clip(img, -1., 1.)
    img = (img + 1.) / 2.
    img = global_contrast_normalize(img.reshape(1, -1) * 255, scale=55.)
    img = prep._gpu_matrix_dot(img - prep.mean_, prep.P_)
    return img.reshape(img_shape)


def post_process(img, prep, img_shape):
    # normalize without contrast_normalize and mean_subtract
    img = prep.inverse(img.reshape(1, -1))[0]
    img /= np.abs(img).max()
    img = np.clip(img, -1., 1.)
    img = (img + 1.) / 2.
    return img.reshape(img_shape)