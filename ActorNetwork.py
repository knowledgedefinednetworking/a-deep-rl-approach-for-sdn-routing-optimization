"""
ActorNetwork.py
"""
__author__ = "giorgio@ac.upc.edu"
__credits__ = "https://github.com/yanpanlau"

from keras.initializations import normal, glorot_normal
from keras.activations import relu
from keras.layers import Dense, Input, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf

from helper import selu


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, DDPG_config):
        self.HIDDEN1_UNITS = DDPG_config['HIDDEN1_UNITS']
        self.HIDDEN2_UNITS = DDPG_config['HIDDEN2_UNITS']

        self.sess = sess
        self.BATCH_SIZE = DDPG_config['BATCH_SIZE']
        self.TAU = DDPG_config['TAU']
        self.LEARNING_RATE = DDPG_config['LRA']
        self.ACTUM = DDPG_config['ACTUM']

        if self.ACTUM == 'NEW':
            self.acti = 'sigmoid'
        elif self.ACTUM == 'DELTA':
            self.acti = 'tanh'

        self.h_acti = relu
        if DDPG_config['HACTI'] == 'selu':
            self.h_acti = selu

        K.set_session(sess)

        #Now create the model
        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_dim):
        S = Input(shape=[state_size], name='a_S')
        h0 = Dense(self.HIDDEN1_UNITS, activation=self.h_acti, init=glorot_normal, name='a_h0')(S)
        h1 = Dense(self.HIDDEN2_UNITS, activation=self.h_acti, init=glorot_normal, name='a_h1')(h0)
        # https://github.com/fchollet/keras/issues/374
        V = Dense(action_dim, activation=self.acti, init=glorot_normal, name='a_V')(h1)
        model = Model(input=S, output=V)
        return model, model.trainable_weights, S
