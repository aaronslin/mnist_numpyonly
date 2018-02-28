import numpy as np
import nn

class FeedForward():
	def __init__(self, arch, dims, lossF, lr):
		self.arch = arch
		self.depth = len(arch)
		self.lossF = lossF()
		self.learning_rate = lr
		self.layers = self.setup_architecture(arch, dims)

		# self.lastInput = [None for i in range(self.depth)]
		# self.lastOutput = [None for i in range(self.depth)]

	def setup_architecture(self, arch, dims):
		layers = []
		previous = None
		for index, layerUnit, dim in zip(range(self.depth), arch, dims):
			# Initialize layer
			layer = layerUnit()
			if layer.hasParams:
				layer.initialize_dimensions(dim)

			# Set parent and child
			layer.parentUnit = previous
			if index != 0:
				previous.childUnit = layer

			previous = layer
			layers.append(layer)

		self.root = layers[0]
		self.tail = layers[-1]
		return layers

	def initialize_weights(self, mode="uniform"):
		for layer in self.layers:
			if layer.hasParams:
				layer.initialize_weights(mode)


	def forward(self, input):
		print input
		output = self.root.forward(input)
		return output
		# lastInput = []
		# lastOutput = []

		# prevInput = input
		# for layer in self.layers:
		# 	lastInput.append(prevInput)
		# 	output = layer.forward(prevInput)
		# 	lastOutput.append(output)
		# 	prevInput = output

		# self.lastInput = lastInput
		# self.lastOutput = lastOutput

	def eval(self, predicted, label):
		evaluation = self.lossF.eval(predicted, label)
		return evaluation

	def backward(self, loss):
		grads = self.lossF.gradient
		self.tail.backprop(grads)

		for layer in self.layers:
			if not layer.hasParams:
				continue
			layer.update_weights(self.learning_rate)


		# prevError = loss
		# gradients = []

		# for i in reversed(range(self.depth)):
		# 	layer = self.layers[i]
		# 	input = self.lastInput[i]

		# 	layer.d_input(prevError, input)

	def clear_history(self):
		for layer in self.layers:
			layer.clear_history()
		self.lossF.zero_grad()



if __name__ == "__main__":
	# Not an elegant input format, but whatever
	# Needs to be checked that these arrays line up
	pass





