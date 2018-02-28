import numpy as np

class NNUnit(object):
	def __init__(self, hasParams):
		self.name="NNUnit"
		self.parentUnit = None
		self.childUnit = None
		self.hasParams = hasParams
		self.clear_history()

		# Stores dLoss/dUnitOutput

	def initialize_dimensions(self, dim):
		(inp_size, out_size) = dim
		self.inp_size = inp_size
		self.out_size = out_size

	def forwardprop(self):
		print "Forward:", self.name, "\t", self.lastInput, "\t", self.lastOutput
		print self.weights if self.hasParams else None
		print "................................................."
		if self.childUnit is not None:
			return self.childUnit.forward(self.lastOutput)
		else: 
			return self.lastOutput

	def clear_history(self):
		self.lastOutput = None
		self.lastInput = None
		self.lastGrad = None

	def print_back(self):
		print "Backward:", self.name, "\t", self.lastInput, "\t", self.lastOutput
		print "\tWeights", self.weights if self.hasParams else None
		print self.lastGrad
		print "................................................."


class Linear(NNUnit):
	def __init__(self):
		super(Linear, self).__init__(hasParams=True)
		self.name="NNUnit.Linear"

	def initialize_weights(self, mode, value=0.1):
		dim = (self.inp_size, self.out_size)
		if mode=="uniform":
			print "Initializing uniform weights!"
			self.weights = np.full(dim, value)
		elif mode=="random":
			self.weights = np.random.normal(size=dim)


	def forward(self, input):
		output = np.matmul(input, self.weights)
		self.lastOutput = output
		self.lastInput = input
		return self.forwardprop()

	def update_weights(self, learning_rate):
		input = self.lastInput
		grad = self.lastGrad
		weight_grads = np.matmul(input.T, grad)
		self.weights = self.weights - learning_rate * weight_grads

	def d_input(self):
		return self.weights.T

	def backprop(self, grads):
		# This code is almost rewritten. Abstract it?
		self.lastGrad = grads
		self.print_back()
		newGrad = np.matmul(grads, self.d_input())
		if self.parentUnit is not None:
			self.parentUnit.backprop(newGrad)

class Relu(NNUnit):
	def __init__(self):
		super(Relu, self).__init__(hasParams=False)
		self.name="NNUnit.Relu"

	def forward(self, input):
		input[input < 0] = 0
		output = np.copy(input)

		self.lastInput = input
		self.lastOutput = output
		return self.forwardprop()

	def d_input(self):
		input = self.lastInput
		return np.array(input > 0).astype(np.float32)

	def backprop(self, grads):
		self.lastGrad = grads
		self.print_back()
		newGrad = grads * self.d_input()
		if self.parentUnit is not None:
			self.parentUnit.backprop(newGrad)


# Implementation of below is paused until Relu is implemented correctly
class Softmax(NNUnit):
	def __init__(self):
		super(Softmax, self).__init__(hasParams = False)
		self.name = "NNUnit.Softmax"

	def forward(self, input):
		expLayer = np.power(np.e, input)
		output = expLayer / np.sum(expLayer)
		self.lastInput = input
		self.lastOutput = output
		return self.forwardprop()


	def d_input(self):
		"""
		grads: batch X nClasses
		Input: batch X nClasses
		Uses the property that dSoftmax(x)/dx = Softmax(x)(1-Softmax(x))
		"""
		out = self.lastOutput
		return (out*(1-out))

	def backprop(self, grads):
		self.lastGrad = grads
		self.print_back()
		newGrad = grads * self.d_input()
		if self.parentUnit is not None:
			self.parentUnit.backprop(newGrad)


class CrossEntropyLoss():
	def __init__(self):
		self.zero_grad()

	def eval(self, pred, label):
		"""
		label is one-hot, should be batch_size X out_size
		pred should be batch_size X out_size
		output is batch_size X 1
		"""
		losses = label * np.log(pred)
		loss = -np.sum(losses, axis=1).reshape((-1,1))
		self.gradient = self.d_pred(loss, label, pred)
		print "CrossEntropyLoss", pred
		print "\t", label
		return loss

	def d_pred(self, loss, label, pred):
		return -label * (1.0/pred) * loss

	def zero_grad(self):
		self.gradient = 0

if __name__ == "__main__":
	X = [Linear, Relu, Linear]
	print [x== Linear for x in X]

	X = [Linear(1, 2), Relu(), Linear(2, 3), Relu()]
	Y = [x.forward(np.array([0])) for x in X]
	Z = [x.getParams() for x in X]
	print Y





