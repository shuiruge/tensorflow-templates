#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
---------

This template makes:

    1. Construct a model via TensorFlow graph. (Herein, examplify this by
       linear regression, which is simple enough for keeping mind clear.)

    2. Run the graph by `tf.Session`.

    3. Save graph and checkpoint.

    4. Run the trained model outside of its training by restore the model from
       the trained checkpoint.
       
It includes:
    
    1. A model template in the `BasicTensorFlowModel`.
    
    2. A general trainer in the `Trainer`, which support the most general
       training process, employing arbitrary collection of data-sets (e.g.
       training data, validation data, testing data, etc).
       
       
HOWTO
------

To use this template, just scan over the code of `BasicTensorFlowModel`, while
taking care of the docstrings (i.e. those "Notes" and inline docstrings).
While scaning, modify those helper blocks as you need. When scaning ends, you
then construct the model you want.

(Note that those "Notes" in docstrings (briefly) illustrate the underlying
principle of TensorFlow, while the inline docstring provides the guide of
modification.)

The `Trainer` is general, thus needs little modification.

And then, imitate the code in the `if __name__ == "__main__"` to run them all.

Thus ends _le travail_.



Preliminaries: Linear Regression
---------

Suppose data_X and data_Y are one-to-one correspondent, we are to find a
linear function `Y = w * X + b` such that it fits the data best. Precisely,
we are to find the values of parameters `w` and `b`, s.t. the cost-function

        def cost_function(weight, bias):
            ''' Square sum cost-fucntion. '''

            def model_output(x):

                return weight * x + bias

            error = [model_output(x) - y
                     for x, y in zip(data_X, data_Y)
                     ]

            return sum(square(error))

is minimized. In addition, we call the `weight` and the `bias`,in according to
the terminology of neural network.
"""


# For compatible with Python2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
script_path = os.path.dirname(os.path.abspath( __file__ ))


# Import TensorFlow
# ---------

# `tensorflow` is generally togethered with `numpy`.
import tensorflow as tf
import numpy as np

from utils import ensure_directory  # `tf.Saver()` cannot save if the directory
                                    # does not exist. Sounds funny.
import matplotlib.pyplot as plt  # to show the result up.





class BasicTensorFlowModel:
    """
    A basic template of optimizational model on TensorFlow, in the style of
    TensorFlow, i.e. stype of graphical. This is a simple linear regression model.
    Even though, it contains all essential materials for models of any complexity.
    
    To use this basic model, you can directly modify the defined helper blocks
    by yourself, to consist with your model.

    Attributes:
        inputs: tf.placeholder
        targets: tf.placeholder
        outputs: tf.Variable (?)
        cost: tf.Variable (?)
        optimizer: tf.train.Optimizer
        summary: tf.summary
    """

    def __init__(self):
        
        self._build_graph()
        

    def _build_graph(self):
        """
        Build up the TensorFlow graph of the `Model`.

        Notes:
            A general algorithm of a optimizational system, e.g. neural network,
            involves several general steps:

                1. placeholders for inputs and targets as a mini-batch;
                2. outputs of the inputs in the mini-batch made by the model;
                3. then cost-function, so as to characterize the distinces
                   between outputs and targets;
                4. optimizer to automatically tune parameters of the model, to
                minimize the cost;
                5. in addition, a summary operation that write the information
                   of training (process and result) into a visualization system.

            All these components are created seperately in the following.
            
            Remarkable that the building of graph and the running of the built
            graph are seperated in TensorFlow. Each helper block, when called,
            **"un-wraps" and then "imprints" its graph** into the whole graph.
            So, as _TensorFlow for Machine Intelligence_ puts, repeatedly calling
            the helper blocks will add new sub-graphs into the whole graph. This
            sounds weired, but a natural implication of the seperation of the
            building and the running of graph.
            
            (This is also called the "lazy property". A `lazy_property`
            decorator, as a magic method, can help reduce the number of codes,
            but we will not use herein, since it makes my mind confused, thus
            non-Pythonic.)
        """
        
        self.graph = tf.Graph()

        with self.graph.as_default():

            self._create_placeholders()
            self._create_outputs()
            self._create_cost()
            self._create_optimizer()
            self._create_summary()


    def _create_placeholders(self):
        """
        Helper block of `self._build_graph()`.

        1. placeholders for inputs and targets as a mini-batch;

        Notes:
            `self.inputs` and `self.targets` are placeholders, since they are
            input, from data outside of the tensorflow-graph when running it
            in a session. Recall that in tensorflow, building and running a
            graph are seperated.

            **After insertion into the placeholders while running in session,
            `self.inputs` and `self.targets` will become `tf.constant`s.**

            Thus, they are not instances of `tf.Variable`, thus will not be
            affected as training by optimizer. See below.
        """
        
        # The `shape` of `self.inputs` and `self.targets` shall be consistent
        # with your model.
        
        # As an instance, herein `shape=[None]` implies a list of floats,
        # corresponding to a mini-batch of scalar data, while leaving the
        # length of the list, as the batch-size, unknown (as `None`).
        
        with tf.name_scope('inputs'):

            self.inputs = tf.placeholder(tf.float32,
                                         shape=[None],
                                         name='inputs'
                                         )

        with tf.name_scope('targets'):

            self.targets = tf.placeholder(tf.float32,
                                          shape=[None],
                                          name='targets'
                                          )



    def _create_outputs(self):
        """
        Helper block of `self._build_graph()`.

        2. outputs of the inputs in the mini-batch made by the model;
        
        Notes:
            `weights` and `biases` are those parameters to be trained, to be
            updated in every epoch of the later training, **by optimizer**.
            Thus they are mutable quantities. In tensorflow, such quantities
            are of `tf.Variable()` class.

            So, they are different from the `self.inputs` and `self.targets`,
            which will not be updated **by optimizer**, as `tf.constant`.

            Instances of `tf.Variable` shall be initialized to some values,
            since these values will be as the starting point of the later
            training by optimizer.

            Herein, we initialize `weights` and `biasis` randomly, by standard
            (truncated) normal distribution.
        """
        
        # While constructing your model (thus its outputs), keep in mind of
        # the consistency of all the shapes of flowing tensors.

        with tf.name_scope('outputs'):

            with tf.name_scope('parameters'):

                weight = tf.Variable(tf.truncated_normal(shape=[1]),
                                      name='weight'
                                      )
                bias = tf.Variable(tf.truncated_normal(shape=[1]),
                                     name='bias'
                                     )


            with tf.name_scope('outputs'):

                outputs = tf.add(bias,
                                 tf.multiply(weight, self.inputs),
                                 name='outputs'
                                 )

            self.outputs = outputs



    def _create_cost(self):
        """
        Helper block of `self._build_graph()`.

        3. then cost-function, so as to characterize the distinces between
           targets and outputs.

        Notes:
            Cross-entropy is encouraged by Nilson, since it increases the
            learning efficiency of gradient based optimizers.
            
            However, for keeping mind clear, herein we use the simplest
            "square sum" as cost.
        """

        with tf.name_scope('cost'):

            with tf.name_scope('square_sum_cost'):
            
                errors = tf.subtract(self.targets, self.outputs,
                                     name='errors'
                                     )
                square_sum_cost = tf.reduce_sum(tf.square(errors))
                
            self.cost = square_sum_cost



    def _create_optimizer(self):
        """
        Helper block of `self._build_graph()`.

        4. optimizer to automatically tune parameters of the model, to
           minimize the cost;

        Notes:
            Runing `optimizer` will update all `tf.Variable` that `cost` depends
            and whose `trainable` flat is `True` (default), s.t. the `cost` is
            minimized. In this case, such `trainable` `tf.Variable`s are `weights`
            and `biases`.

            Indeed, notice that any other `tf.Variable`, i.e. `self.outputs`,
            is completely restricted by other `tf.Variable`s and `tf.constant`s
            (will be input via the two placeholders).
            
            The dependences are shown explicity in `tensorboard`.
        """

        with tf.name_scope('optimizer'):
            
            # Generally `learning_rate` is regarded as a model parameter.
            learning_rate = 0.01
            
            # `Adam` optimizer is the best amoung the gradient based optimizers.
            self.optimizer = \
                tf.train.AdamOptimizer(learning_rate).minimize(self.cost)



    def _create_summary(self):
        """
        Helper block for `self._build_graph()`.

        5. in addition, a summary operation that write the information of
           training (process and result) into a visualization system.

        Notes:
            To `tensorboard`.

            There are many blocks in `tensorboard`. However, herein we only
            add `SCALAR` and `HISTOGRAM` blocks in `tensorboard`, which
            visualizes the evoluation of our training by the plot and histogram
            of `self.cost`.
        """

        with tf.name_scope('summary'):

            tf.summary.scalar("cost", self.cost)
            tf.summary.histogram("histrogram_cost", self.cost)
            # etc.

            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary = tf.summary.merge_all()


    # Thus Finished the construction of TensorFlow Graph.




class Trainer:
    """
    Train the model by feeding training, vaidation, testing data, etc.


    Args:
        model:
            Can be any optimizational model on tensorflow (as class) as long as
            involving the following attributes:
                model.graph, model.inputs, model.targets,
                model.optimizer, model.summary.
                
        batch_generators: [Generator]
            Each generator yield a mini-batch of data for training, validation,
            or testing, respectively. We assume that the `batch_generators[0]`
            is that for training data, which will be used by `self.optimizer`.
            
        max_training_steps: int
            The training will start at a restored checkpoint, as its inital step,
            the practical maximum of training steps is the sum of the initial
            step and the `max_training_steps`.
            
        skip_step: int
            While training, the trainer will save the checkpoint for every
            `skip_step` steps.
            
        path_to_checkpoint: str
            Path to the `checkpoint` (*.ckpt) file which restore (if exists)
            or save checkpoint of training. It shall be of the form:
                
                '<path_to_checkpoints_dir>/<model_name>.ckpt'.
                
        path_to_graph: str
            Path to the `logdir` of `tensorboard`, in which the training summary
            of the model is to be saved.
    """
    
    def __init__(self,
                 model,
                 batch_generators,
                 max_training_steps,
                 skip_step,
                 path_to_graph,
                 path_to_checkpoint
                 ):
        
        self.model = model
        self.batch_generators = batch_generators
        self.max_training_steps = max_training_steps
        self.skip_step = skip_step
        self.path_to_graph = path_to_graph
        self.path_to_checkpoint = path_to_checkpoint  
        
        self._sess = tf.Session(graph=self.model.graph)
        


    def train(self):
        """ Train the model. """
        
        with self._sess:

            self._prepare_for_training()
            
            training_steps = range(self._initial_step,
                                   self.max_training_steps
                                   )
            for step in training_steps:
    
                self._train_by_feeding(step)
                
                if (step + 1) % self.skip_step == 0:
                    
                    self._save_checkpoint(step)
                
                else:
                    pass
            
            self._postpare_for_training()
        



    def _prepare_for_training(self):
        """
        General setup of preparing for a training of TensorFlow model.
        
        Explicitly:
            1. create writers for each data-set (training, validation, testing,
               etc);
            
            2. initialize global step, which keep track of checkpoint;
            
            3. create saver;
            
            4. initialize all `tf.Variable`s in the model.
            
            5. get the latest checkpoint. if exists, then continue the training
               from the latest checkpoint.
        """
        
        # Create writer for each data-set
        # (i.e. training, validation, and testing, etc).
        self._writers = []
        for i, _ in enumerate(self.batch_generators):
            writer = tf.summary.FileWriter(self.path_to_graph, self._sess.graph)
            self._writers.append(writer)

        # global_step to keep track of checkpoint
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        
        # Create saver
        self._saver = tf.train.Saver()
        
        # Initialize all `tf.Variable`s in one go
        self._sess.run(tf.global_variables_initializer())
        
        # Get checkpoint
        # CAUTION that the arg of `get_checkpoint_state` is `checkpoint_dir`,
        # i.e. the directory of the `checkpoint` to be restored from.
        ckpt = tf.train.get_checkpoint_state(
                os.path.dirname(self.path_to_checkpoint)
                )
        self._initial_step = 0

        # If that checkpoint exists, then restore from the checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            
            self._saver.restore(self._sess,ckpt.model_checkpoint_path)
            
            # A rude way of reading the step of the latest checkpoint.
            # And assign it as the initial step of the later training.
            self._initial_step = \
                int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
            
        else:
            pass
        
        
        

    def _train_by_feeding(self, global_step):
        """
        Train the model by feeding the data (`model.inputs` and `model.targets`
        from `batch_generators`.
        
        Args:
            global_step: int
        """
        
        for i, batch_generator in enumerate(self.batch_generators):
            
            inputs, targets = next(batch_generator)
            
            feed_dict = {  model.inputs: inputs,
                           model.targets: targets
                           }
            
            if i == 0:  
                # Meaning that it's training data. For training data, we shall
                # update the parameters (i.e. `trainable` `tf.Variable`s)in
                # self.model.
                # (Recall that we have demanded to place the batch-generator of
                #  training data to the first place of the list of generators.)
                self._sess.run(model.optimizer,
                               feed_dict=feed_dict
                               )
            else:
                pass
            
            # Write to `tensorboard`.
            summary = self._sess.run(model.summary,
                                     feed_dict=feed_dict
                                     )
            self._writers[i].add_summary(summary,
                                         global_step=global_step
                                         )
            

            

    def _save_checkpoint(self, global_step):
        """
        Args:
            global_step: int
        """

        # `tf.saver` cannot automatically mkdir, so
        ensure_directory(self.path_to_checkpoint)
        
        self._saver.save(self._sess,
                         self.path_to_checkpoint,
                         global_step=global_step
                         )
        
    
    
    def _postpare_for_training(self):
        """
        General setup of postparing for a training of TensorFlow model.
        
        Explicitly:
            1. write the training summaries to disk (for `tensorboard`);
            
            2. close writers
            
            3. close session.
        """
        
        # While ending:
        for writer in self._writers:
            
            # Write the summaries to disk
            writer.flush()
            
            # Close the SummaryWriter
            writer.close()
            
        # Close the session
        self._sess.close()




if __name__ == "__main__":
    """
    We will do the followings:
        
        1. generate dummy data from an arbitrarily given `target fuction`;
        2. construct the batch generators which emit different kinds of data,
           for training, validation, or testing, etc;
        3. set up and train the model with mini-batches emited from batch
           generators;
        4. restore the model and compute the outputs (predictions) as the result
           of the training;
        5. plot the result of training out.
    """
    
    # Parameters
    # ---------
    path_to_graph = os.path.join(script_path,
                                 '../dat/graphs/'
                                 )
    path_to_checkpoint = os.path.join(script_path,
                                      '../dat/checkpoints/basic_tf_mod.ckpt'
                                      )
    
    
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
            (data_X: np.array(float),
             data_Y: np.array(float)
             )
            where len(data_X) = len(data_Y) = X_size.
        """
    
        data_X = np.linspace(start=0, stop=10, num=X_size)
    
        data_Y = np.array(
            [target_function(x) + 0.7 * random.gauss(0, 1) for x in data_X]
            )
    
        return (data_X, data_Y)
    
    
    X_size = 100
    data_X, data_Y = generate_dummy_data(X_size)
    
    
    
    # Construct Batch Generators
    # ---------
    
    def batch_generator(data_X=data_X, data_Y=data_Y, batch_size=10):
        """
        Args:
            data_X: np.array(float)
            data_Y: np.array(float)
            batch_size: int
        Yield:
            (np.array(float), np.array(float))
            with the two arraies the same length, as `batch_size`.
        """
        
        while True:
            
            xs = []
            ys = []
            
            for i in range(batch_size):
        
                p = random.randint(0, X_size - 1)
                
                x = data_X[p]
                y = data_Y[p]
                
                xs.append(x)
                ys.append(y)
                
            batch = (np.array(xs), np.array(ys))
            
            yield batch
    
    
    batch_generators = []
    
    for i in range(2):
        
        bg = batch_generator()
        
        batch_generators.append(bg)
        
    
    
    # Set up and Train the Model
    # ---------
    
    model = BasicTensorFlowModel()
    
    trainer = Trainer(model,
                      batch_generators,
                      max_training_steps=10 ** 3,
                      skip_step=10,
                      path_to_graph=path_to_graph,
                      path_to_checkpoint=path_to_checkpoint
                      )
    trainer.train()
    
    
    # Compute Outputs (Predictions) from the Trained Model
    # ---------
    
    with tf.Session(graph=model.graph) as sess:
            
        saver = tf.train.Saver()
        
        # Restore the trained model from its checkpoint
        # CAUTION that the arg of `get_checkpoint_state` is `checkpoint_dir`,
        # i.e. the directory of the `checkpoint` to be restored from.
        ckpt = tf.train.get_checkpoint_state(
                   os.path.dirname(path_to_checkpoint)
                   )
        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            
        best_fit_Y = sess.run(model.outputs,
                              feed_dict={model.inputs: data_X}
                              )
        
    
    # Plot the Result
    # ---------
    
    plt.plot(data_X, data_Y, '.')
    plt.plot(data_X, best_fit_Y, '--')
    plt.show()