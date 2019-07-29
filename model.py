from sys import stdout
from typing import List

from util import *


class Model:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        # number of features
        num_features = 10304  # 112 x 92
        # number of target labels
        num_labels = 40
        # learning rate (alpha)
        learning_rate = 0.1

        # input data
        self.train_dataset = train_data
        self.train_labels = train_labels
        self.test_dataset = test_data
        self.test_labels = test_labels

        # initialize a tensorflow graph
        self.graph = tf.Graph()

        with self.graph.as_default():
            """ 
            defining all the nodes 
            """
            # Inputs
            self.tf_train_dataset = tf.placeholder(tf.float32, shape=(None, num_features))
            self.tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
            # tf_valid_dataset = tf.constant(valid_dataset)
            self.tf_test_dataset = tf.constant(test_data)

            # Variables.
            weights = tf.Variable(tf.random.truncated_normal([num_features, num_labels]))
            biases = tf.Variable(tf.zeros([num_labels]))
            # Output layer  / Training computation. Logits = raw predictions (wo softmax
            logits = tf.matmul(self.tf_train_dataset, weights) + biases
            # loss = cross-entropy
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.tf_train_labels, logits=logits))
            # Optimizer.
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

            # Predictions for the training, and test data.
            self.train_prediction = tf.nn.softmax(logits)
            self.test_prediction = tf.nn.softmax(tf.matmul(self.tf_test_dataset, weights) + biases)

            # Output predictions
            self.predictions = tf.argmax(logits, 1)

            # performance metrics
            correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(self.tf_train_labels, 1))
            self.accuracy_measure = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            self.sess_init = tf.compat.v1.global_variables_initializer()

        self.session = tf.Session(graph=self.graph)

    def test(self, test_x, test_y)-> float:
        return (self.session.run(self.accuracy_measure, feed_dict={self.tf_train_dataset: test_x, self.tf_train_labels: test_y}))

    def predict(self, data) -> List[float]:
        return self.predictions.eval(session=self.session, feed_dict={self.tf_train_dataset: [data]})

    def train_gd(self, epochs, acc_limit=0.9, disp_freq=250):
        # with self.session as session:
        # initialize weights and biases
        self.session.run(self.sess_init)
        print("Initialized")
        for i in range(epochs + 1):
            feed_dict = {self.tf_train_dataset: self.train_dataset,
                         self.tf_train_labels: self.train_labels}
            self.session.run(self.optimizer, feed_dict=feed_dict)
            if (i % disp_freq == 0):
                train_acc = self.test(self.train_dataset, self.train_labels)
                test_acc = self.test(self.test_dataset, self.test_labels)
                print(f"Train Acc. : {train_acc} \t Test Acc. : {test_acc}")
                if test_acc > acc_limit:
                    print(f"Reached cutoff at {acc_limit:.3f}; halting after {i} iterations")
                    break
        print("\n")

    def train_sgd(self, num_steps, acc_limit=0.9, disp_freq=500, batch_size = 128):
        # with self.session as session:
        # initialize weights and biases
        self.session.run(self.sess_init)
        print("Initialized")
        for step in range(num_steps):
            # pick a randomized offset
            offset = np.random.randint(0, self.train_labels.shape[0] - self.batch_size - 1)

            # Generate a minibatch.
            batch_data = self.train_dataset[offset:(offset + batch_size), :]
            batch_labels = self.train_labels[offset:(offset + batch_size), :]

            # Prepare the feed dict
            feed_dict = {self.tf_train_dataset: batch_data,
                         self.tf_train_labels: batch_labels}

            # run one step of computation
            _, l, predictions = self.session.run([self.optimizer, self.loss, self.train_prediction],
                                            feed_dict=feed_dict)

            if (step % disp_freq == 0):
                print(f"Minibatch loss at step {step}: {l}")
                print(f"Minibatch accuracy: {accuracy(predictions, batch_labels):.1}")
                test_acc = self.test(self.test_dataset, self.test_labels)
                if  test_acc > acc_limit:
                    print(f"Reached {test_acc:.3f} acc, cutoff at {acc_limit:.3f}; halting after {step} iterations")
                    break

        print(f"\nTest accuracy: {self.test(self.test_dataset, self.test_labels):.3f}")


    def invert(self, session, num_iters, lam, img, pre_process, pred_cutoff=0.9999, disp_freq=50):
        """
        :param session: tf.session
        :param num_iters: alpha parameter, iterations to apply GD. 5000 in papaer
        :param lam: lambda parameter, gradient step size. 0.1 in paper
        :param img: img to invert???
        :param pre_process: zca object
        :param pred_cutoff:
        :param disp_freq:
        :return:
        """

        # probabilities = self.preds(img, session) # [1][40] float
        probabilities = session.run(self.probabilities, feed_dict={self.x: [img]}) # [1][40] float
        class_ind = session.run(self.class_inds, feed_dict={self.x: [img]})[0] # this is the class we are inverting
        current_X = np.zeros(list(img.shape)[0]).astype(np.float32) # 10304 (pixel count)
        Y = (one_hot_preds(probabilities)).astype(np.float32) # [40] in {0,1}
        best_X = np.copy(current_X)
        best_loss = 100000.0
        prev_losses = [100000.0] * 100

        for i in range(num_iters):
            feed_dict = {self.x: [current_X], self.y_: Y}
            der, current_loss = session.run([self.grads, self.loss], feed_dict)
            # todo loss= 0, fix that
            # der:List[ndarray] size = 10304

            # image manipulation
            current_X = np.clip(current_X - lam * (der[0][0]), 0.0, 1.0)
            #                   x_i - lambda * gradient(x_i)
            #   why der[0][0] ?? why not each pixel its corresponding gradient????
            # len(der[0][0])=10304, that's why
            # current_X = normalize(current_X, pre_process, current_X.shape)
            # todo
            # current_X = equalize(np.reshape(current_X, (112, 92)))
            # current_X = sharpenFilter(current_X)
            # current_X = np.ndarray.flatten(current_X)

            probabilities = session.run(self.probabilities, feed_dict={self.x: [img]})[0]

            if i % 50 == 49:
                # todo every how many iterations?
                # todo randomly? apply sharpening and gaussian filters, edge filters?
                # current_X = sharpenFilter(np.reshape(current_X, (112, 92)))
                current_X = gaussianBlur(np.reshape(current_X, (112, 92)), 8)
                current_X = equalize(current_X)
                current_X = np.ndarray.flatten(current_X)

            if current_loss < best_loss:
                best_loss = current_loss
                best_X = current_X

            if current_loss > 2 * max(prev_losses):
                print("\n Breaking due to gradient chaos!!")
                break

            if pred_cutoff < probabilities[class_ind]:
                print(f"\n Above Probability Criteria!: {probabilities[class_ind]}")
                print(f"\n After {i} iterations")
                print(f"\n {probabilities}")
                break

            if i % disp_freq == 0:
                #                 plt.close()
                #                 face_imshow(post_process(current_X, pre_process, current_X.shape))
                #                 plt.show()
                stdout.write(f"\r Acc: {probabilities[class_ind]} and Loss: {current_loss} and Best Loss: {best_loss}")
                stdout.flush()

        stdout.write("\n")
        print('Loop Escape.')

        current_preds = session.run(self.probabilities, feed_dict={self.x: [current_X]})
        best_preds = session.run(self.probabilities, feed_dict={self.x: [best_X]})
        current_X = post_process(current_X, pre_process, current_X.shape)
        best_X = post_process(best_X, pre_process, best_X.shape)
        return current_X, current_preds, best_X, best_preds

    def close(self):
        self.session.close()