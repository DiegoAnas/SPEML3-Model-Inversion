from pylearn2.datasets.preprocessing import ZCA
from pylearn2.expr.preprocessing import global_contrast_normalize
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from util import *
from model import Model


def perform_inversion(pre_process: bool, images):
    for img in images:
        face_imshow(img)
        plt.title('Image-Class used for inversion.')
        plt.show()
        print('Predictions: ' + str((model.preds(img))))

        inv_img_last, inv_img_last_p, inv_img_best, inv_img_best_p = model.invert(session, 100, 0.1, img,
                                                                                  pre_process= pre_process)

        face_imshow(inv_img_best)
        plt.title('Best Image after inversion.')
        plt.show()
        print('Predictions: ' + str(inv_img_best_p))

        face_imshow(inv_img_last)
        plt.title('Last Iteration Image after inversion.')
        plt.show()
        print('Predictions: ' + str(inv_img_last_p))

if __name__ == "__main__":

    #Load dataset
    train_x, test_x, train_y, test_y  = unpack_facedataset() # 7:3 ratio for train:test

    # preprocess images
    # GCN and ZCA object!!
    ##Normalize and then ZCA whitening
    ## Normalized data only used on inversion, not in training
    # train_x_normalized = global_contrast_normalize(train_x * 255, scale=55.)
    # zca = ZCA()
    # zca.fit(train_x_normalized)

    x = tf.placeholder(tf.float32, shape=[None, 112*92])
    y_ = tf.placeholder(tf.float32, shape=[None, 40])
    model = Model(x,y_)
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    model.train(train_x, train_y, session, test_x, test_y, 250)

    # perform_inversion(zca, test_x[0::3])
    perform_inversion(train_x, test_x[0::3])
