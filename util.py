import random
import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image, ImageOps, ImageFilter
from pylearn2.expr.preprocessing import global_contrast_normalize
from keras import backend as K
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import matplotlib.pyplot as plt


def weight_variable(shape):
    # Outputs random values from a truncated normal distribution.
    initial = tf.random.truncated_normal(shape=shape, stddev=0.1)
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


def unpack_facedataset(path='./DATA/att_faces/orl_faces'
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
                    # Convert to grayscale (Floyed-SteinBerg-dithering)
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
    train_x, test_x, train_y, test_y = train_x.reshape(40 * 7, 112 * 92),\
                                       test_x.reshape(40 * 3,112 * 92),\
                                       train_y.reshape(40 * 7),\
                                       test_y.reshape(40 * 3)

    return train_x, test_x, one_hot_class(train_y), one_hot_class(test_y)


def normalize(img, prep, img_shape):
    # this requires zca from pylearn 2 for all functions prep.
    # img = prep.inverse(img.reshape(1, -1))[0]

    imgarr = np.zeros((1, 1, 112, 92), dtype='float32')
    imgarr[0][0] = img.reshape(112, 92)
    img = prep.flow(imgarr, batch_size=1)
    img = img[0]

    img /= np.abs(img).max()
    img = np.clip(img, -1., 1.)
    img = (img + 1.) / 2.

    img = global_contrast_normalize(img.reshape(1, -1) * 255, scale=55.)
    # img = prep._gpu_matrix_dot(img - prep.mean_, prep.P_)
    return img.reshape(img_shape)

def post_process(img, prep, img_shape):
    # normalize without contrast_normalize and mean_subtract
    # this requires zca from pylearn 2 for all functions prep.
    # img = prep.inverse(img.reshape(1, -1))[0]
    imgarr = np.zeros((1, 1, 112, 92), dtype='float32')
    imgarr[0][0] = img.reshape(112, 92)
    img = prep.flow(imgarr)[0]

    img /= np.abs(img).max()
    img = np.clip(img, -1., 1.)
    img = (img + 1.) / 2.
    return img.reshape(img_shape)


def perform_inversion(model, person_class, option:int=0, equalize:bool=False, filter_freq=1):
    filters = []
    if option == 1:
        filters.append(ImageFilter.GaussianBlur(2))
        filters.append(ImageFilter.SHARPEN)
    if option == 2:
        filters.append(ImageFilter.BLUR)
        filters.append(ImageFilter.SHARPEN)
    if option == 3:
        filters.append(ImageFilter.GaussianBlur(2))
    if option == 4:
        filters.append(ImageFilter.SHARPEN)
    if option == 6:
        filters.append(ImageFilter.SHARPEN)
        filters.append(ImageFilter.GaussianBlur(2))
    if option == 5:
        filters.append(ImageFilter.DETAIL)
    if option == 7:
        filters.append(ImageFilter.MedianFilter(size=3))
        filters.append(ImageFilter.SHARPEN)
    if option == 8:
        filters.append(ImageFilter.MedianFilter(size=5))
        filters.append(ImageFilter.SHARPEN)
    # inv_img_last, inv_img_last_p, inv_img_best, inv_img_best_p =\
    #     model.invert_by_pixel(person_class, filters, equalize, lambda_=1, epochs=3, iterations=2,
    #                           filter_freq=filter_freq)

    inv_img_last, inv_img_last_p, inv_img_best, inv_img_best_p = \
        model.invert(person_class, filters, equalize, lambda_=0.2, iterations=5000,
                              filter_freq=filter_freq)

    face_imshow(inv_img_best)
    plt.title('Best Image after inversion.')
    plt.show()
    print('Best Predictions: ' + str(inv_img_best_p))

    face_imshow(inv_img_last)
    plt.title('Last Iteration Image after inversion.')
    plt.show()
    print('Last Predictions: ' + str(inv_img_last_p))


def perform_inversion2(pre_process, images, model, session):
    for img in images:
        face_imshow(img)
        plt.title('Image-Class used for inversion.')
        plt.show()
        print('Predictions: ' + str((model.preds(img, session))))

        inv_img_last, inv_img_last_p, inv_img_best, inv_img_best_p = model.invert(session, 5000, 0.1, img,
                                                                                  pre_process= pre_process)

        face_imshow(inv_img_best)
        plt.title('Best Image after inversion.')
        plt.show()
        print('Best Predictions: ' + str(inv_img_best_p))

        face_imshow(inv_img_last)
        plt.title('Last Iteration Image after inversion.')
        plt.show()
        print('Last Predictions: ' + str(inv_img_last_p))


def load(file_name):
    with open(file_name, 'rb') as pickle_file:
        return pickle.load(pickle_file)


def save(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

# utility function to calculate accuracy
def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (correctly_predicted) / predictions.shape[0]
    return accu

#  Pillow library functions for image modificatoins
def identity(img_array):
    """
    Test function
    :param img_array:
    :return: same array after converting it to PIL Image
    """
    pil_img = Image.fromarray((img_array * 255).astype('uint8'), mode='L')
    new_array = np.array(pil_img) / 255
    return new_array

def applyEqualization(img_array):
    """
    :param img_array: a 2D numpy array
    :return: numpy array after applying histogram equalization
    """
    pil_img = Image.fromarray((img_array * 255).astype('uint8'), mode='L')
    pil_img = ImageOps.equalize(pil_img)
    return np.array(pil_img)/255


def applyFilter(img_array, filter:ImageFilter):
    pil_img = Image.fromarray((img_array * 255).astype('uint8'), mode='L')
    pil_img = pil_img.filter(filter)
    return np.array(pil_img)/255


def gaussianBlur(img_array, radius:float=2):
    """
    :param img_array: a 2D numpy array
    :param radius: Blur radius
    :return: numpy array after applying a blur filter
    """
    pil_img = Image.fromarray((img_array * 255).astype('uint8'), mode='L')
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius))
    return np.array(pil_img)


def medianFilter(img_array: np.array, size: int = 3) -> np.array:
    """
    :param img_array: a 2D numpy array
    :param size: window size
    :return: numpy array after applying a median filter
    """
    pil_img = Image.fromarray((img_array * 255).astype('uint8'), mode='L')
    pil_img = pil_img.filter(ImageFilter.MedianFilter(size))
    return np.array(pil_img)

def showPil(img_array, title):
    pil_img = Image.fromarray((img_array * 255).astype('uint8'), mode='L')
    pil_img.show(title=title)



