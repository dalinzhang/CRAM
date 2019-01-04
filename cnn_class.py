#! /usr/bin/python3

import tensorflow as tf

class cnn:
	def __init__(
			self,
			weight_stddev	= 0.1,
			bias_constant	= 0.1,
			padding			= "SAME",
			):
			self.weight_stddev	= weight_stddev
			self.bias_constant	= bias_constant
			self.padding		= padding

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = self.weight_stddev)
		return tf.Variable(initial)


	def bias_variable(self, shape):
		initial = tf.constant(self.bias_constant, shape = shape)
		return tf.Variable(initial)


	def conv1d(self, x, W, kernel_stride):
	# API: must strides[0]=strides[4]=1
		return tf.nn.conv1d(x, W, stride=kernel_stride, padding=self.padding)


	def conv2d(self, x, W, kernel_stride):
	# API: must strides[0]=strides[4]=1
		return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding=self.padding)


	def conv3d(self, x, W, kernel_stride):
	# API: must strides[0]=strides[4]=1
		return tf.nn.conv3d(x, W, strides=[1, kernel_stride, kernel_stride, kernel_stride, 1], padding=self.padding)


	def apply_conv1d(self, x, filter_width, in_channels, out_channels, kernel_stride, train_phase):
		weight = self.weight_variable([filter_width, in_channels, out_channels])
		bias = self.bias_variable([out_channels]) # each feature map shares the same weight and bias
		conv_1d = tf.add(self.conv1d(x, weight, kernel_stride), bias)
		conv_1d_bn = self.batch_norm_cnv_1d(conv_1d, train_phase)
		return tf.nn.elu(conv_1d_bn)


	def apply_conv2d(self, x, filter_height, filter_width, in_channels, out_channels, kernel_stride, train_phase):
		weight = self.weight_variable([filter_height, filter_width, in_channels, out_channels])
		bias = self.bias_variable([out_channels]) # each feature map shares the same weight and bias
		conv_2d = tf.add(self.conv2d(x, weight, kernel_stride), bias)
		conv_2d_bn = self.batch_norm_cnv_2d(conv_2d, train_phase)
		return tf.nn.elu(conv_2d_bn)


	
	def apply_conv3d(self, x, filter_depth, filter_height, filter_width, in_channels, out_channels, kernel_stride, train_phase):
		weight = self.weight_variable([filter_depth, filter_height, filter_width, in_channels, out_channels])
		bias = self.bias_variable([out_channels]) # each feature map shares the same weight and bias
		conv_3d = tf.add(self.conv3d(x, weight, kernel_stride), bias)
		conv_3d_bn = self.batch_norm_cnv_3d(conv_3d, train_phase)
		return tf.nn.elu(conv_3d_bn)


	def batch_norm_cnv_3d(self, inputs, train_phase):
		return tf.layers.batch_normalization(inputs, axis=4, momentum=0.993, epsilon=1e-5, scale=False, training=train_phase)


	def batch_norm_cnv_2d(self, inputs, train_phase):
		return tf.layers.batch_normalization(inputs, axis=3, momentum=0.993, epsilon=1e-5, scale=False, training=train_phase)


	def batch_norm_cnv_1d(self, inputs, train_phase):
		return tf.layers.batch_normalization(inputs, axis=2, momentum=0.993, epsilon=1e-5, scale=False, training=train_phase)


	def batch_norm(self, inputs, train_phase):
		return tf.layers.batch_normalization(inputs, axis=1, momentum=0.993, epsilon=1e-5, scale=False, training=train_phase)


	def apply_max_pooling(self, x, pooling_height, pooling_width, pooling_stride):
	# API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
		return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1], strides=[1, pooling_stride, pooling_stride, 1], padding=self.padding)


	def apply_max_pooling3d(self, x, pooling_depth, pooling_height, pooling_width, pooling_stride):
	# API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
		return tf.nn.max_pool3d(x, ksize=[1, pooling_depth, pooling_height, pooling_width, 1], strides=[1, pooling_stride, pooling_stride, pooling_stride, 1], padding=self.padding)

	
	def apply_fully_connect(self, x, x_size, fc_size, train_phase):
		fc_weight = self.weight_variable([x_size, fc_size])
		fc_bias = self.bias_variable([fc_size])
		fc = tf.add(tf.matmul(x, fc_weight), fc_bias)
		fc_bn = self.batch_norm(fc, train_phase)
		return tf.nn.elu(fc_bn)

	
	def apply_readout(self, x, x_size, readout_size):
		readout_weight = self.weight_variable([x_size, readout_size])
		readout_bias = self.bias_variable([readout_size])
		return tf.add(tf.matmul(x, readout_weight), readout_bias)
