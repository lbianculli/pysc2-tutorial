import os
import numpy as np
import tensorflow as tf
import pysc2
from pysc2.lib import actions, features


class A3CAgent():
	''' Proto agent for training on minigames. 
	Main methods are build_net, step, and update'''
  	def __init__(self, training, mm_size, screen_size, screen_weights, mm_weights, biases, name='A3C'):
		self.name = name
		self.training = training
		self.summary = []  
		assert mm_size == screen_size  # paper recommended to begin, but not ideal
		self.mm_size = mm_size
		self.screen_size = screen_size
		self.info_size = len(actions.FUNCTIONS)  # ?
		self.screen_weights = screen_weights
		self.mm_weights = mm_weights
		self.biases = biases
		self.screen_channels = 17
		self.mm_channels = 7
		
		
	def setup(self, sess, summary_writer):
		self.sess = sess
		self.summary_writer = summary_writer
		
	
	def initialize_agent(self):
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		
	
	def reset(self):
		# E schedule
		self.epsilon = [0.05, 0.2]
		
	
	def build_network(self, reuse, dev,):
		with tf.variable_scope(self.name) and tf.device(dev):
			
			# set network inputs -- placeholders, weights, biases, info
			self.screen = tf.placeholder(tf.float32, [None, self.screen_size, self.screen_size, self.screen_channels], name='screen')
			self.minimap = tf.placeholder(tf.float32, [None, self.mm_size, self.mm_size, self.screen_channels], name='mm')
			self.info = tf.placeholder(tf.float32, [None, self.info_size], name='info')  # still dont really get
			
			# build network
			conv_net = create_network(self.screen_channels, self.mm_channels,  # this needs to be cleaned i think
									  self.screen_weights, self.mm_weights, self.biases, len(actions.FUNCTIONS))
			
			self.spatial_action, self.non_spatial_action, self.value = net
			
			# set targets
			
			
			# compute probs
			
			
	
