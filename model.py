from sys import stdout
from typing import List

from util import *


class SoftmaxModel:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        # number of features
        self.num_features = 10304  # 112 x 92
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
            self.tf_train_dataset = tf.compat.v1.placeholder(tf.float32, shape=(None, self.num_features))
            self.tf_train_labels = tf.compat.v1.placeholder(tf.float32, shape=(None, num_labels))
            # tf_valid_dataset = tf.constant(valid_dataset)
            self.tf_test_dataset = tf.constant(test_data)

            # Variables.
            weights = tf.Variable(tf.random.truncated_normal([self.num_features, num_labels], stddev=0.1))
            # weights = tf.Variable(tf.random.truncated_normal([self.num_features, num_labels]))
            # biases = tf.Variable(tf.zeros([num_labels]))
            biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))
            # Output layer  / Training computation. Logits = raw predictions (wo softmax
            logits = tf.matmul(self.tf_train_dataset, weights) + biases
            # loss = cross-entropy
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.tf_train_labels, logits=logits))
            # Gradient descent
            self.grads = tf.gradients(self.cross_entropy, self.tf_train_dataset)
            # Optimizer.
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(self.cross_entropy)

            # Predictions for the training, and test data.
            self.probabilities = tf.nn.softmax(logits)
            self.test_prediction = tf.nn.softmax(tf.matmul(self.tf_test_dataset, weights) + biases)

            self.class_inds = tf.argmax(self.probabilities, 1)  # class predicted
            class_ind_correct = tf.argmax(self.tf_train_labels, 1)  # true class
            self.class_prob = (self.probabilities[0, tf.cast(class_ind_correct[0], tf.int32)])  # Prob predicted of the true class
            self.loss = tf.abs(tf.subtract(tf.constant(0.99), self.class_prob))  # loss is 1 - prob of true class
            ###

            # Output predictions
            self.predictions = tf.argmax(logits, 1)

            # performance metrics
            correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(self.tf_train_labels, 1))
            self.accuracy_measure = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            # initializer
            self.sess_init = tf.compat.v1.global_variables_initializer()

        self.session = tf.compat.v1.Session(graph=self.graph)

    def test(self, test_x, test_y)-> float:
        return (self.session.run(self.accuracy_measure, feed_dict={self.tf_train_dataset: test_x, self.tf_train_labels: test_y}))

    def predict(self, data) -> List[float]:
        return self.predictions.eval(session=self.session, feed_dict={self.tf_train_dataset: [data]})

    def get_probabilities(self, data) -> List[float]:
        return self.probabilities.eval(session=self.session, feed_dict={self.tf_train_dataset: [data]})

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
            offset = np.random.randint(0, self.train_labels.shape[0] - batch_size - 1)

            # Generate a minibatch.
            batch_data = self.train_dataset[offset:(offset + batch_size), :]
            batch_labels = self.train_labels[offset:(offset + batch_size), :]

            # Prepare the feed dict
            feed_dict = {self.tf_train_dataset: batch_data,
                         self.tf_train_labels: batch_labels}

            # run one step of computation
            _, l, predictions = self.session.run([self.optimizer, self.cross_entropy, self.probabilities],
                                            feed_dict=feed_dict)

            if (step % disp_freq == 0):
                print(f"Minibatch loss at step {step}: {l}")
                print(f"Minibatch accuracy: {accuracy(predictions, batch_labels):.1}")
                test_acc = self.test(self.test_dataset, self.test_labels)
                if  test_acc > acc_limit:
                    print(f"Reached {test_acc:.3f} acc, cutoff at {acc_limit:.3f}; halting after {step} iterations")
                    break

        print(f"\nTest accuracy: {self.test(self.test_dataset, self.test_labels):.3f}")


    def invert(self, person_class, filters=[], equalize=False, lambda_=0.1, iterations=1000, disp_freq=500,
               pred_cutoff=0.9, filter_freq=5):
        # current_X = np.full(list([10304])[0], 0.5, dtype=float) # 10304 (pixel count)
        current_X = np.random.rand(10304).astype(np.float32)  # 10304 (pixel count)
        Y = np.zeros([40])  # [40] in {0,1}
        Y[person_class] = 1
        best_X = np.copy(current_X)
        best_loss = 100000.0
        prev_losses = [100000.0] * 100
        face_imshow(current_X)
        plt.title('Initial')
        plt.show()
        end = False
        for i in range(iterations):
            feed_dict = {self.tf_train_dataset: [current_X], self.tf_train_labels: [Y]}
            gradients, current_loss = self.session.run([self.grads, self.loss], feed_dict)
            # gradients:List[ndarray] size = 10304

            # pixel_indices = list(range(i%2, len(current_X), 2))
            current_X = np.clip(current_X - (lambda_ * (gradients[0][0])), 0.0, 1.0)
            # plt.show()
            if (len(filters) > 0 or equalize) and i % filter_freq == filter_freq-1:
                current_X = np.reshape(current_X, (112, 92))
                for filter in filters:
                    # showPil(current_X, "before")
                    current_X = applyFilter(current_X, filter)
                    # showPil(current_X, "after")
                if equalize:
                    current_X = applyEqualization(current_X)
                current_X = np.ndarray.flatten(current_X)

            probabilities = self.session.run(self.probabilities, feed_dict={self.tf_train_dataset: [current_X]})[0]
            if current_loss < best_loss:
                best_loss = current_loss
                best_X = current_X
                best_iteration = i
            if current_loss > 2 * max(prev_losses):
                print("\n Breaking due to gradient chaos!!")
                break
            # if probabilities[person_class] > pred_cutoff and current_loss < 1-pred_cutoff:
            #     print(f"\n Above Probability Criteria!: {probabilities[person_class]}")
            #     print(f"\n After {i} iterations w loss {current_loss}")
            #     print(f"\n {probabilities}")
            #     break
            if i % disp_freq == 0 and not end:
                # face_imshow(current_X)
                # plt.title(f'It {i}')
                # plt.show()
                # end = len(input("\nShow more results?"))>0
                print(f"\n Acc: {probabilities[person_class]} and Loss: {current_loss} and Best Loss: {best_loss}")
            # if i % 555 == 0 and not end:
            #     random.shuffle(current_X)

        print('Loop Escape.')
        print(f"Best found at {best_iteration}")
        current_preds = self.session.run(self.probabilities, feed_dict={self.tf_train_dataset: [current_X]})
        best_preds = self.session.run(self.probabilities, feed_dict={self.tf_train_dataset: [best_X]})
        # current_X = post_process(current_X, pre_process, current_X.shape)
        # best_X = post_process(best_X, pre_process, best_X.shape)
        return current_X, current_preds, best_X, best_preds

    def invert_by_pixel(self, person_class, filters=[], equalize=False, lambda_=0.1, epochs=5, iterations=5000, disp_freq=500,
               pred_cutoff=0.9, filter_freq=5):
        # current_X = np.full(list([10304])[0], 0.5, dtype=float) # 10304 (pixel count)
        current_X = np.random.rand(10304).astype(np.float32)  # 10304 (pixel count)
        Y = np.zeros([40])  # [40] in {0,1}
        Y[person_class] = 1
        best_X = np.copy(current_X)
        best_loss = 100000.0
        prev_losses = [100000.0] * 100
        face_imshow(current_X)
        plt.title('Initial')
        plt.show()
        new_X = np.zeros([10304], dtype=float)
        end = False
        for epoch in range(epochs):
            for pixel in range(self.num_features):
                for i in range(iterations):
                    feed_dict = {self.tf_train_dataset: [current_X], self.tf_train_labels: [Y]}
                    gradients = self.session.run(self.grads, feed_dict)
                    # gradients:List[ndarray] size = 10304

                    # pixel_indices = list(range(i%2, len(current_X), 2))
                    current_X[pixel] = np.clip(current_X[pixel] - (lambda_ * (gradients[0][0][pixel])), 0.0, 1.0)

                    # if (len(filters) > 0 or equalize) and i % filter_freq == 0:
                    #     current_X = np.reshape(current_X, (112, 92))
                    #     for filter in filters:
                    #         # showPil(current_X, "before")
                    #         current_X = applyFilter(current_X, filter)
                    #         # showPil(current_X, "after")
                    #     if equalize:
                    #         current_X = applyEqualization(current_X)
                    #     current_X = np.ndarray.flatten(current_X)

                    # if probabilities[person_class] > pred_cutoff and current_loss < 1-pred_cutoff:
                    #     print(f"\n Above Probability Criteria!: {probabilities[person_class]}")
                    #     print(f"\n After {i} iterations w loss {current_loss}")
                    #     print(f"\n {probabilities}")
                    #     break
                    # if i % disp_freq == 0 and not end:
                    #     face_imshow(current_X)
                    #     plt.title(f'It {i}')
                    #     plt.show()
                    #     end = len(input("\nShow more results?"))>0
                    #     stdout.write(f"\r Acc: {probabilities[person_class]} and Loss: {current_loss} and Best Loss: {best_loss}")
                    #     stdout.flush()
            current_loss = self.session.run(self.loss, feed_dict)
            if current_loss < best_loss:
                best_loss = current_loss
                best_X = current_X
            if current_loss > 2 * max(prev_losses):
                print("\n Breaking due to gradient chaos!!")
                break
            probabilities = self.session.run(self.probabilities, feed_dict={self.tf_train_dataset: [current_X]})[0]
            print (f"\r Acc: {probabilities[person_class]} and Loss: {current_loss} and Best Loss: {best_loss}")
            face_imshow(current_X)
            plt.title(f'Image after {epoch} epochs')
            plt.show()
            face_imshow(best_X)
            plt.title(f'Best Image after {epoch} epochs')
            plt.show()

        stdout.write("\n")
        print('Loop Escape.')
        current_preds = self.session.run(self.probabilities, feed_dict={self.tf_train_dataset: [current_X]})
        best_preds = self.session.run(self.probabilities, feed_dict={self.tf_train_dataset: [best_X]})
        # current_X = post_process(current_X, pre_process, current_X.shape)
        # best_X = post_process(best_X, pre_process, best_X.shape)
        return current_X, current_preds, best_X, best_preds


    def close(self):
        self.session.close()

