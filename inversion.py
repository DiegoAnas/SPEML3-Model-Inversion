from pylearn2.datasets.preprocessing import ZCA
from pylearn2.expr.preprocessing import global_contrast_normalize
from util import *
from model import Model


if __name__ == "__main__":

    #Load dataset
    train_x, test_x, train_y, test_y  = unpack_facedataset() # 7:3 ratio for train:test

    # preprocess images
    # GCN and ZCA object!!
    # Normalize and then ZCA whitening
    # Normalized data only used on inversion, not in training
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

    perform_inversion(zca, test_x[0::3], model, session)
    # stride 3 sicne there are 3 faces per ecah class in test set
