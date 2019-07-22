from sys import stdout
from util import *

class Model:
    def __init__(self, x, y_):
        in_dim = int(x.get_shape()[1])  # 10304 for Face dataset
        out_dim = int(y_.get_shape()[1])  # 40 for Face dataset
        self.x = x
        # switiching to a simple 2-layer network with relu
        W = weight_variable([in_dim, out_dim])
        b = bias_variable([out_dim])
        self.y = tf.matmul(x, W) + b  # output layer
        self.probabilities = tf.nn.softmax(self.y)
        self.class_inds = tf.argmax(self.probabilities, 1)  ## class indices????

        #cross_entropy?
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=self.y))

        class_ind_correct = tf.argmax(y_, 1)
        self.class_prob = (self.probabilities[0, tf.cast(class_ind_correct[0], tf.int32)])
        self.loss = tf.subtract(tf.constant(1.0), self.class_prob)

        # Gradient descent
        self.grads = tf.gradients(self.cross_entropy, x)

        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, train_x, train_y, session, test_x, test_y, num_iters, disp_freq=50):
        for i in range(num_iters):
            feed_dict = {self.x: train_x, self.y_: train_y}
            session.run(self.train_step, feed_dict)
            if (i % disp_freq == 0):
                train_acc = self.test(train_x, train_y, session)
                test_acc = self.test(test_x, test_y, session)
                stdout.write("\r Train Acc. : %f    Test Acc. : %f" % (train_acc, test_acc))
                stdout.flush()
        stdout.write("\n")

    def test(self, test_x, test_y, session):
        return (session.run(self.accuracy, feed_dict={self.x: test_x, self.y_: test_y}))

    def invert(self, session, num_iters, lam, img, pre_process, pred_cutoff=0.99, disp_freq=1):

        probabilities = self.preds(img, session)
        # replace it w?:
        # probabilities = session.run(self.probabilities, feed_dict={self.x: [img]})
        class_ind = session.run(self.class_inds, feed_dict={x: [img]})[0]
        current_X = np.zeros(list(img.shape)[0]).astype(np.float32)
        Y = (one_hot_preds(probabilities)).astype(np.float32)
        best_X = np.copy(current_X)
        best_loss = 100000.0
        prev_losses = [100000.0] * 100

        for i in range(num_iters):
            feed_dict = {self.x: [current_X], self.y_: Y}
            der, current_loss = session.run([self.grads, self.loss], feed_dict)
            current_X = np.clip(current_X - lam * (der[0][0]), 0.0, 1.0)
            current_X = normalize(current_X, pre_process, current_X.shape)
            probabilities = self.preds(current_X)[0]

            if current_loss < best_loss:
                best_loss = current_loss
                best_X = current_X

            if current_loss > 2 * max(prev_losses):
                print("\n Breaking due to gradient chaos!!")
                break

            if pred_cutoff < probabilities[class_ind]:
                print("\n Above Probability Criteria!: {0}".format(probabilities[class_ind]))
                break

            if i % disp_freq == 0:
                #                 plt.close()
                #                 face_imshow(post_process(current_X, pre_process, current_X.shape))
                #                 plt.show()
                stdout.write("\r Acc: %f and Loss: %f and Best Loss: %f" % (probabilities[class_ind], current_loss, best_loss))
                stdout.flush()

        stdout.write("\n")
        print('Loop Escape.')

        current_preds = self.preds(current_X)
        best_preds = self.preds(best_X)
        current_X = post_process(current_X, pre_process, current_X.shape)
        best_X = post_process(best_X, pre_process, best_X.shape)
        return current_X, current_preds, best_X, best_preds

    def preds(self, img, session): ## what does this do?
        """
        :param img:
        :param session:
        :return:
        """
        return session.run(self.probabilities, feed_dict={self.x: [img]})