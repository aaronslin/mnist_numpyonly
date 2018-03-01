import numpy as np
import nn

class FeedForward():
	"""
	General class for feedforward networks. 

	Assumes that each layer, except the first and last, has one layer feeding in 
	(the parent layer) it and feeds out into another layer (child layer). Although
	this architecture is not very flexible, the code can easily be adapted to
	suit any DAG architecture by allowing multiple parents/children and summing gradients.

	Architecture specification
	---------------------------
	arch: the order of layers in the feed-forward network
	dims: the dimensions of (input, output) in layers with trainable parameters
			(i.e. only the NNUnit.Linear layers in the current implementation)
	lossFunc: the loss function to be used
	"""
	def __init__(self, arch, dims, lossFunc):
		self.arch = arch
		self.depth = len(arch)
		self.lossFunc = lossFunc()
		self.layers = self.setup_architecture(arch, dims)

	def setup_architecture(self, arch, dims):
		layers = []
		previous = None
		for index, layerUnit, dim in zip(range(self.depth), arch, dims):
			# Initialize layer
			layer = layerUnit()
			if layer.hasParams:
				layer.initialize_dimensions(dim)

			# Set parent and child of layer
			layer.parentUnit = previous
			if index != 0:
				previous.childUnit = layer

			previous = layer
			layers.append(layer)

		self.root = layers[0]
		self.tail = layers[-1]
		return layers

	def initialize_weights(self, mode):
		for layer in self.layers:
			if layer.hasParams:
				layer.initialize_weights(mode)


	def forward(self, input):
		# Forward-pass an input through the network
		output = self.root.forward(input)
		return output

	def eval(self, predicted, label):
		evaluation = self.lossFunc.eval(predicted, label)
		return evaluation

	def backward(self, loss, learnRate):
		# Calculate the gradients, starting in the back
		grads = self.lossFunc.gradient
		self.tail.backprop(grads)

		# Update the weights
		for layer in self.layers:
			if not layer.hasParams:
				continue
			layer.update_weights(learnRate)

	def clear_history(self):
		for layer in self.layers:
			layer.clear_history()
		self.lossFunc.zero_grad()



