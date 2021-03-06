import numpy as np
import theano
import theano.tensor as T

"""
to define the layer like this:
l1 = Layer(inputs, in_size = 1, out_size = 10, activation_function)
l2 = Layer(l1.outputs, 10, 1, None)

"""
class Layer(object):
	
	def __init__(self, inputs, in_size, out_size, activation_function=None):
		self.W = theano.shared(np.random.normal(0,1,(in_size,out_size)))
		self.b = theano.shared(np.zeros((out_size,)) + 0.1)
		self.Wx_plus_b = T.dot(inputs, self.W) + self.b
		self.activation_function = activation_function
		if activation_function is None:
			self.outputs = self.Wx_plus_b
		else:
			self.outputs = self.actvation_function(self.Wx_plus_b)
