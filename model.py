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
            self.loss = tf.abs(tf.subtract(tf.constant(1.0), self.class_prob))  # loss is 1 - prob of true class
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
        max_gradients= []
        end = False
        for i in range(iterations):
            feed_dict = {self.tf_train_dataset: [current_X], self.tf_train_labels: [Y]}
            gradients, current_loss = self.session.run([self.grads, self.loss], feed_dict)
            # gradients:List[ndarray] size = 10304

            current_X = np.clip(current_X - (lambda_ * (gradients[0][0])), 0.0, 1.0)
            if i < 20:
                max_gradients.append(max(gradients[0][0]))
            if (len(filters) > 0 or equalize) and i % filter_freq == filter_freq-1:
                current_X = np.reshape(current_X, (112, 92))
                for filter in filters:
                    current_X = applyFilter(current_X, filter)
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

        print('Loop Escape.')
        print(f"Best found at {best_iteration}")
        current_preds = self.session.run(self.probabilities, feed_dict={self.tf_train_dataset: [current_X]})
        best_preds = self.session.run(self.probabilities, feed_dict={self.tf_train_dataset: [best_X]})
        print(max_gradients)
        plt.plot(range(len(max_gradients)), max_gradients, label="Maximum value")
        plt.xticks(range(len(max_gradients)))
        plt.xlabel("Iteration")
        plt.ylabel("Gradient value")
        plt.show()
        return current_X, current_preds, best_X, best_preds

    def close(self):
        self.session.close()


class MLPModel:
    def __init__(self, train_data, train_labels, test_data, test_labels):
        # number of features
        self.num_features = 10304  # 112 x 92
        # number of target labels
        num_labels = 40
        # learning rate (alpha)
        starting_learning_rate = 0.01
        #
        regularizer_rate = 0.1
        num_layers_0 = 3000

        # input data
        self.train_dataset = train_data
        self.train_labels = train_labels
        self.test_dataset = test_data
        self.test_labels = test_labels

        ## for dropout layer
        keep_prob = tf.placeholder(tf.float32)

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

            # Layers
            # Input layer
            weights_0 = tf.Variable(tf.random.truncated_normal([self.num_features, num_layers_0],
                                                             stddev=(1 / tf.sqrt(float(self.num_features)))))
            bias_0 = tf.Variable(tf.random_normal([num_layers_0]))
            # sigmoid or not??
            hidden_output_0 = tf.nn.sigmoid(tf.matmul(self.tf_train_dataset, weights_0) + bias_0)

            # Hidden layer
            weights_1 = tf.Variable(tf.random.truncated_normal([num_layers_0, num_labels],
                                                               stddev=(1 / tf.sqrt(float(self.num_features)))))
            bias_1 = tf.Variable(tf.random_normal([num_labels]))
            self.predicted_y = tf.nn.sigmoid(tf.matmul(hidden_output_0, weights_1) + bias_1)

            #
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predicted_y, labels=self.tf_train_labels)) \
                   + regularizer_rate * (tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)))

            ## Variable learning rate
            learning_rate = tf.train.exponential_decay(starting_learning_rate, 0, 5, 0.85, staircase=True)

            ## Metrics definition
            correct_prediction = tf.equal(tf.argmax(self.tf_train_labels, 1), tf.argmax(self.predicted_y, 1))
            self.accuracy_measure = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # self.loss2 = tf.subtract(tf.constant(1.0), self.accuracy_measure)

            ## Optimzer for finding the right weight
            # optimizer = tf.train.AdamOptimizer(learning_rate).\
            #     minimize(loss, var_list=[weights_0, weights_1, bias_0, bias_1])
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate). \
                minimize(self.loss, var_list=[weights_0, weights_1, bias_0, bias_1])
            self.gradients = tf.gradients(self.loss, self.tf_train_dataset)
            # gradient_step = optimizer.compute_gradients

            # initializer
            self.sess_init = tf.compat.v1.global_variables_initializer()
            self.saver = tf.train.Saver()

        self.session = tf.compat.v1.Session(graph=self.graph)

    def test(self, test_x, test_y)-> float:
        return (self.session.run(self.accuracy_measure, feed_dict={self.tf_train_dataset: test_x, self.tf_train_labels: test_y}))

    def predict(self, data) -> List[float]:
        return self.predictions.eval(session=self.session, feed_dict={self.tf_train_dataset: [data]})

    def get_probabilities(self, data) -> List[float]:
        return self.probabilities.eval(session=self.session, feed_dict={self.tf_train_dataset: [data]})

    def train_gd(self, epochs, acc_limit=0.9, disp_freq=50):
        #134.55644391700116 seconds for 1000 iters
        # .74 train and test acc
        # with self.session as session:
        # initialize weights and biases
        try:
            self.saver.restore(self.session, "./MLPModel.ckpt")
            print("Model restored.")
        except:
            self.session.run(self.sess_init)
            print("Initialized")
            training_accuracy = []
            training_loss = []
            testing_accuracy = []
            for i in range(epochs):
                feed_dict = {self.tf_train_dataset: self.train_dataset,
                             self.tf_train_labels: self.train_labels}
                _, acc ,cost = self.session.run([self.optimizer, self.accuracy_measure, self.loss], feed_dict=feed_dict)
                training_accuracy.append(acc)
                training_loss.append(cost)
                ## Evaluation of model
                testing_accuracy.append(self.test(self.test_dataset, self.test_labels))
                if (i % disp_freq == 0):
                    print(f"Train Acc. : {acc} \t training loss : {cost}")
                    print(f"Testing Acc. : {testing_accuracy[i]} ")
                    # if test_acc > acc_limit:
                    #     print(f"Reached cutoff at {acc_limit:.3f}; halting after {i} iterations")
                    #     break
            print("\n")
            save_path = self.saver.save(self.session, "./MLPModel.ckpt")
            print("Model saved in path: %s" % save_path)


    def train_sgd(self, epochs, acc_limit=0.9, disp_freq=50, batch_size = 200):
        # with self.session as session:
        # initialize weights and biases
        #  111.83265602700158 seconds batch 200
        #  0.025 acc
        self.session.run(self.sess_init)
        print("Initialized")
        training_accuracy = []
        training_loss = []
        testing_accuracy = []
        for i in range(epochs):
            # pick a randomized offset
            offset = np.random.randint(0, self.train_labels.shape[0] - batch_size - 1)

            # Generate a minibatch.
            batch_data = self.train_dataset[offset:(offset + batch_size), :]
            batch_labels = self.train_labels[offset:(offset + batch_size), :]

            # Prepare the feed dict
            feed_dict = {self.tf_train_dataset: batch_data,
                         self.tf_train_labels: batch_labels}
            _, acc ,cost = self.session.run([self.optimizer, self.accuracy_measure, self.loss], feed_dict=feed_dict)
            training_accuracy.append(acc)
            training_loss.append(cost)
            ## Evaluation of model
            testing_accuracy.append(self.test(self.test_dataset, self.test_labels))
            if (i % disp_freq == 0):
                print(f"Train Acc. : {acc} \t training loss : {cost}")
                print(f"Testing Acc. : {testing_accuracy[i]} ")
                # if test_acc > acc_limit:
                #     print(f"Reached cutoff at {acc_limit:.3f}; halting after {i} iterations")
                #     break
        print("\n")

    def invert(self, person_class, filters=[], equalize=False, lambda_=0.1, iterations=1000, disp_freq=100,
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
        max_gradients= []
        end = False
        best_iteration = 0
        delta = 0.01
        for i in range(iterations):
            feed_dict = {self.tf_train_dataset: [current_X], self.tf_train_labels: [Y]}
            gradients, current_loss = self.session.run([self.gradients, self.loss], feed_dict)
            # for pixel in range(self.num_features):
            #     current_X2 = current_X
            #     current_X2[pixel] += delta
            #     feed_dict = {self.tf_train_dataset: [current_X2], self.tf_train_labels: [Y]}
            #     loss2 = self.session.run(self.loss, feed_dict)
            #     gradients[pixel] = (loss2 - current_loss ) / delta
            current_X = np.clip(current_X - (lambda_ * (gradients[0][0])), 0.0, 1.0)
            if i < 20:
                max_gradients.append(max(gradients[0][0]))
            if (len(filters) > 0 or equalize) and i % filter_freq == filter_freq-1:
                current_X = np.reshape(current_X, (112, 92))
                for filter in filters:
                    current_X = applyFilter(current_X, filter)
                if equalize:
                    current_X = applyEqualization(current_X)
                current_X = np.ndarray.flatten(current_X)

            probabilities = self.session.run(self.predicted_y, feed_dict={self.tf_train_dataset: [current_X]})[0]
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
                print(f"\n Acc: {probabilities[person_class]} and Loss: {current_loss} and Best Loss: {best_loss}")

        print('Loop Escape.')
        print(f"Best found at {best_iteration}")
        current_preds = self.session.run(self.predicted_y, feed_dict={self.tf_train_dataset: [current_X]})
        best_preds = self.session.run(self.predicted_y, feed_dict={self.tf_train_dataset: [best_X]})
        print(max_gradients)
        plt.plot(range(len(max_gradients)), max_gradients, label="Maximum value")
        plt.xticks(range(len(max_gradients)))
        plt.xlabel("Iteration")
        plt.ylabel("Gradient value")
        plt.show()
        return current_X, current_preds, best_X, best_preds

    def close(self):
        self.session.close()
