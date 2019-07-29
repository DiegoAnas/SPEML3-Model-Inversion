from util import *
from model import Model

import time
import sys

if __name__ == "__main__":

    #Load dataset
    train_x, test_x, train_y, test_y  = unpack_facedataset() # 7:3 ratio for train:test

    model = Model(train_x, train_y, test_x, test_y)
    start = time.perf_counter()
    model.train_sgd(1000)
    end = time.perf_counter()
    print(f"Training done, {end-start} seconds ellapsed")
    #here a validation set could be tested
    print(f"Test accuracy after training {model.test(test_x, test_y)}")
    print(f"Prediction for test 3 {model.predict(test_x[6])}")
    print(f"Truth for test 3 {test_y[6]}")
    face_imshow(test_x[3])
    plt.title('one pic.')
    plt.show()
    option = 0
    while option != 9:
        option = int(input("option (9=quit): "))
        eq = int(input("eq 0/1: ")) == 1
        perform_inversion(model, 2, option=option, equalize=eq)
    # perform_inversion(model, test_x[0])
    model.close()



