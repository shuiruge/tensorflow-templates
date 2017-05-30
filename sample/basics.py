"""
Description
---------

This template includes:

    1. Construct a graph of linear regression;

    1. Use `tf.placeholder`, `tf.Variable`, "scope", "FileWriter"

    1. Run the graph by `tf.Session`.

    1. Some docstrings are inserted where illustrations are called for.


Linear Regression
---------

Suppose data_X and data_Y are one-to-one correspondent, we are to find a
linear function `Y = w * X + b` such that it fits the data best. Precisely,
we are to find the values of parameters `w` and `b`, s.t. the cost-function

        def cost_function(w, b):

            def predicted(x):

                return w * x + b

            squared_errors = [(predict(x) - y) ** 2
                              for x, y in zip(data_X, data_Y)
                              ]

            return sum(squared_errors)

is minimized. In addition, we call the `w` "weight" and the `b` "bias",
according to the terminology neural network.


Training
------

For keeping template simple, in the training, we will not employ mini-batch
in training. Instead, we will update parameters by running optimizer by one
time for all data_X and data_Y, and repeat several times.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



# Import TensorFlow
# ---------

## `tensorflow` is generally togethered with `numpy`.

import tensorflow as tf
import numpy as np


## In addition, to show up the result:

import matplotlib.pyplot as plt




# Basics Work-Flow
# ---------

## **TensorFlow is generally employed to do optimization.**
##
## So, we are to define some performance and maximize it. Or, some
## "cost (-function)" (e.g. -1 * performance) and minimize it.


## 1. Construct TensorFlow graph, with optimizer that minimize some
##    "cost-function".
## 2. Initialize all instances of `tf.Variable`
## 3. Run optimizer.





# Construct Graph
# ---------

## Initialize a tf.Graph instance

my_graph = tf.Graph()

with my_graph.as_default():



    with tf.name_scope('Data_Inputs'):

        ## Input `X` and `Y`, as placeholders, since they are input, from `data_X` and
        ## `data_Y` respectively. Thus they cannot have explicit values until starting
        ## training, in the running of an instance of `tf.Session`. However, since, in
        ## tensorflow, graph and session are seperated, such `X` and `Y` can be nothing
        ## but as placeholders.

        ## **After input, `X` and `Y` will become `tf.constant`s.**

        X = tf.placeholder(tf.float32, name='X')
        Y = tf.placeholder(tf.float32, name='Y')




    with tf.name_scope('Parameters_to_be_Trained'):

        ## `w` and `b` are those parameters to be trained, to be updated in every epoch
        ## of the later training, **by optimizer**. Thus they are different from the `X`
        ## and `Y`, which will not be updated **by optimizer**.

        ## Notice that `tf.Variable` (with capitial "V") is a class, different from
        ## `tf.placeholder`, as the `X` and `Y`.

        ## Initialize `w` and `b` as `0.0`.

        w = tf.Variable(0.0, name='weight')
        b = tf.Variable(0.0, name='bias')




    with tf.name_scope('Cost'):

        # `tf.multiply` and `tf.add` are element-wise, thus `X` can be of any shape.

        z = tf.multiply(w, X)
        predict = tf.add(z, b)

        errors = tf.subtract(Y, predict, name='errors')
        cost = tf.reduce_sum(
                   tf.square(errors), name='cost'
                   )

        ## In this way, the `z`, `predict`, `errors`, and `cost` are automatically
        ## an instance of `tf.Variable`.




    with tf.name_scope('Optimizer'):

        ## Runing `optimizer` will update all `tf.Variable` that `cost` depends
        ## and whose `trainable` flat is `True` (default), s.t. the `cost` is
        ## minimized. In this case, such `tf.Variable`s are `w` and `b`.
        ## Indeed, notice that any other `tf.Variable`, i.e. `predict`, is
        ## completely restricted by other `tf.Variable`s and `tf.constant`s
        ## (will be input via the two placeholders).

        ## `Adam` is the best amoung the gradient based optimizers.

        learning_rate = 0.1

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



    ## Thus Finished constructing the TensorFlow graph.




# Digression: Generate Dummay Data
# ---------

target_w = 2
target_b = -1

target_function = lambda x: target_w * x + target_b


import random

def generate_dummy_data(X_size):
    """
    Args:
        X_size: int
    Returns:
        (data_X: np.array(float), data_Y: np.array(float))
        where len(data_X) = len(data_Y) = X_size.
    """

    data_X = np.linspace(start=0, stop=10, num=X_size)

    data_Y = np.array(
        [target_function(x) + 0.7 * random.gauss(0, 1) for x in data_X]
        )

    return (data_X, data_Y)


X_size = 100
data_X, data_Y = generate_dummy_data(X_size)





# Run a Session
# ---------

with tf.Session(graph=my_graph) as sess:


    # Initialize `tf.Variable`
    # ---------

    ## Before running `tf.Session` on `optimizer`, all `tf.Variable`s needed by
    ## `optimizer` shall be initialized.

    ## `tf.global_variables_initializer` initializes all `tf.Variable`s in one go.

    initializer = tf.global_variables_initializer()
    sess.run(initializer)



    # Training by Optimiser
    # ---------

    ## As said, for keeping template simple, in the training, we will not employ
    ## mini-batch in training. Instead, we will update parameters by running
    ## optimizer by one time for all data_X and data_Y, and repeat several times.

    training_times = 100

    for step in range(training_times):

        sess.run(optimizer,
                feed_dict={X: data_X, Y: data_Y}
                )



    # Return the Best-Fits
    # ---------

    ## Since `w`, `b`, as instances of `tf.Variable`, are trainable (default),
    ## after running optimizer, they are updated to the best fitted values, s.t.
    ## the `cost` are (almost) minimized.

    w_fitted, b_fitted = sess.run([w, b])




# Plot the Result
# ---------

def best_fit_function(x):

    return w_fitted * x + b_fitted


best_fit_Y = [best_fit_function(x) for x in data_X]
target_Y = [target_function(x) for x in data_X]

plt.plot(data_X, data_Y, '.')
plt.plot(data_X, best_fit_Y, '--')
plt.plot(data_X, target_Y, '-')
#plt.legend()
plt.show()
