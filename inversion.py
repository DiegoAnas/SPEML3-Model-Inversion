from pylearn2.datasets.preprocessing import ZCA
from pylearn2.expr.preprocessing import global_contrast_normalize
from util import *
from model import Model

# ZCA whitening
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
K.set_image_dim_ordering('th')


if __name__ == "__main__":

    #Load dataset
    train_x, test_x, train_y, test_y  = unpack_facedataset() # 7:3 ratio for train:test

    # preprocess images
    # GCN and ZCA object!!
    # Normalize and then ZCA whitening
    # Normalized data only used on inversion, not in training

    # reshape to be [samples][pixels][width][height]
    X_train = train_x.reshape(train_x.shape[0], 1, 112, 92)
    X_test = test_x.reshape(test_x.shape[0], 1, 112, 92)
    # convert from int to float
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # define data preparation
    # datagen = ImageDataGenerator(zca_whitening=True)
    datagen = ImageDataGenerator(featurewise_std_normalization=True, zca_whitening=True)
    # fit parameters from data
    datagen.fit(X_train)
    for X_batch, y_batch in datagen.flow(X_train, train_y, batch_size=9):
        # create a grid of 3x3 images
        for i in range(0, 9):
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(X_batch[i].reshape(112, 92), cmap=pyplot.get_cmap('gray'))
        # show the plot
        pyplot.show()
        break


    try:
        zca = load("faces/zca.data")
    except Exception as e:
        print("Failed to load preprocessed data from disk, computing zca")
        train_x_normalized = global_contrast_normalize(train_x * 255, scale=55.)
        zca = ZCA()
        zca.fit(train_x_normalized)
        save("faces/zca.data", zca)

    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 112*92])
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 40])
    model = Model(x,y_)
    session = tf.compat.v1.InteractiveSession()
    session.run(tf.compat.v1.global_variables_initializer())
    model.train(train_x, train_y, session, test_x, test_y, 250)

    perform_inversion(datagen, test_x[0::3], model, session)
    # stride 3 sicne there are 3 faces per ecah class in test set
