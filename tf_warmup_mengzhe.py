# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range

import numpy as np
import tensorflow as tf
import q_learning


class SGDRegressor:
  def __init__(self, D):
    print("Hello TensorFlow!")
    lr = 10e-2

    # create inputs, targets, params
    # matmul doesn't like when w is 1-D
    # so we make it 2-D and then flatten the prediction

    # self.w = tf.Variable(tf.random_normal(shape=(D, 1)), name='w')
    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

    # make prediction and cost
    # Y_hat = tf.reshape( tf.matmul(self.X, self.w), [-1] )
    # delta = self.Y - Y_hat
    # cost = tf.reduce_sum(delta * delta)


    node_layer_1 = 1
    node_layer_2 = 1
    node_layer_3 = 1


    self.W1 = tf.Variable(tf.random_normal(shape=(D, node_layer_1)), name='w1')
    # self.b1 = tf.Variable(tf.random_normal(shape=(1, )), name='b1')
    self.W2 = tf.Variable(tf.random_normal(shape=(node_layer_1, node_layer_2)), name='w2')
    # self.b2 = tf.Variable(tf.random_normal(shape=(1, )), name='b2')
    # self.W3 = tf.Variable(tf.random_normal(shape=(node_layer_2, node_layer_3)), name='w3')
    # self.b3 = tf.Variable(tf.random_normal(shape=(1, )), name='b3')

    # define the model
    # self.Z1 = tf.nn.relu(tf.matmul(self.X, self.W1) + self.b1)
    # self.Z2 = tf.nn.relu(tf.matmul(self.Z1, self.W2) + self.b2)
    # self.Z3 = tf.matmul(self.Z2, self.W3) + self.b3

    # self.Z1 = tf.nn.relu(tf.matmul(self.X, self.W1))
    # self.Z2 = tf.nn.relu(tf.matmul(self.Z1, self.W2))
    # self.Z3 = tf.matmul(self.Z2, self.W3)

    self.Z1 = tf.matmul(self.X, self.W1)
    self.Z2 = tf.matmul(self.Z1, self.W2)


    Y_hat = tf.reshape(self.Z2, [-1] ) 
    # Y_hat = self.Z1
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
  q_learning.SGDRegressor = SGDRegressor
  q_learning.main()
