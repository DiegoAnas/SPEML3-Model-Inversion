from util import *
from model import Model

import time

if __name__ == "__main__":

    #Load dataset
    train_x, test_x, train_y, test_y  = unpack_facedataset() # 7:3 ratio for train:test

    model = Model(train_x, train_y, test_x, test_y)
    start = time.perf_counter()
    model.train_sgd(7000)
    end = time.perf_counter()
    print(f"Training done, {end-start} seconds ellapsed")
    #here a validation set could be tested
    print(f"Test accuracy after training {model.test(test_x, test_y)}")
    print(f"Prediction for test 0 {model.predict(test_x[0])}")
    print(f"Truth for test 0 {test_y[0]}")
    face_imshow(test_x[0])
    plt.title('one pic.')
    plt.show()
    model.close()



