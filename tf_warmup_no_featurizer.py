# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range

import numpy as np
import tensorflow as tf
import q_learning_no_featurizer

## this implements a shallow neural network with single layer for Q learning
# the performance is not good
# lack of experience replay and a target network

class SGDRegressor:
  def __init__(self, D):
    print("Hello TensorFlow!")
    lr = 10e-2

    # create inputs, targets, params
    # matmul doesn't like when w is 1-D
    # so we make it 2-D and then flatten the prediction

    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

    node_layer_1 = 1
    node_layer_2 = 5
    node_layer_3 = 1

    self.W1 = tf.Variable(tf.random_normal(shape=(D, node_layer_1)), name='w1')
    # self.b1 = tf.Variable(tf.random_normal(shape=(1, )), name='b1')
    # self.W2 = tf.Variable(tf.random_normal(shape=(node_layer_1, node_layer_2)), name='w2')
    # self.b2 = tf.Variable(tf.random_normal(shape=(1, )), name='b2')
    # self.W3 = tf.Variable(tf.random_normal(shape=(node_layer_2, node_layer_3)), name='w3')
    # self.b3 = tf.Variable(tf.random_normal(shape=(1, )), name='b3')

    # define the model
    self.Z1 = tf.matmul(self.X, self.W1) 
    # self.Z2 = tf.nn.relu(tf.matmul(self.Z1, self.W2) + self.b2)
    # self.Z3 = tf.matmul(self.Z2, self.W3) + self.b3


    # make prediction and cost
    Y_hat = tf.reshape(self.Z1, [-1] )     
    delta = self.Y - Y_hat
    cost = tf.reduce_sum(delta * delta)

    # ops we want to call later
    # self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    self.train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    self.predict_op = Y_hat
  
    # start the session and initialize params
    init = tf.global_variables_initializer()
    self.session = tf.InteractiveSession()
    self.session.run(init)

  def partial_fit(self, X, Y):
    self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

  def predict(self, X):
    return self.session.run(self.predict_op, feed_dict={self.X: X})


if __name__ == '__main__':
  q_learning_no_featurizer.SGDRegressor = SGDRegressor
  q_learning_no_featurizer.main()
