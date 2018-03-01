import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
DEBUG_MODE = False

class NNUnit(object):
	"""
	General class for NN layers. Implements forwardprop and backprop abstractly, and expects
	instances to implement a d_input() method needed for backprop.
	"""
	def __init__(self, hasParams):
		self.name="NNUnit"
		self.parentUnit = None
		self.childUnit = None
		self.hasParams = hasParams
		self.clear_history()

	def initialize_dimensions(self, dim):
		(inp_size, out_size) = dim
		self.inp_size = inp_size
		self.out_size = out_size

	def clear_history(self):
		self.lastOutput = None
		self.lastInput = None

		# Stores dLoss/dUnitOutput, not dLoss/dUnitInput
		self.lastGrad = None

	def forwardprop(self):
		self.print_forwardprop()
		if self.childUnit is not None:
			return self.childUnit.forward(self.lastOutput)
		else: 
			return self.lastOutput

	def print_forwardprop(self):
		if DEBUG_MODE:
			print "Forward:", self.name, "\t", self.lastInput, "\t", self.lastOutput
			print self.weights if self.hasParams else None
			print "................................................."

	def backprop(self, grads):
		self.lastGrad = grads
		self.print_backprop()
		newGrad = self.d_input(grads)
		if self.parentUnit is not None:
			self.parentUnit.backprop(newGrad)

	def print_backprop(self):
		if DEBUG_MODE:
			print "Backward:", self.name, "\t", self.lastInput, "\t", self.lastOutput
			print self.lastGrad
			print "................................................."

	def d_input(self):
		raise NotImplemented


class Linear(NNUnit):
	"""
	Implementation of a fully-connected linear unit. These are the only layers
	currently implemented that have trainable parameters. 

	Further modifications can be done to include bias weights.
	"""
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

	def update_weights(self, learningRate):
		input = self.lastInput
		grad = self.lastGrad
		weight_grads = np.matmul(input.T, grad)
		self.weights = self.weights - learningRate * weight_grads

	def d_input(self, grads):
		chain = self.weights.T
		return np.matmul(grads, chain)

class Relu(NNUnit):
	"""
	Implements a RELU.
	"""
	def __init__(self):
		super(Relu, self).__init__(hasParams=False)
		self.name="NNUnit.Relu"

	def forward(self, input):
		input[input < 0] = 0
		output = np.copy(input)

		self.lastInput = input
		self.lastOutput = output
		return self.forwardprop()

	def d_input(self, grads):
		input = self.lastInput
		chain = np.array(input > 0).astype(np.float32)
		return grads * chain

class Softmax(NNUnit):
	"""
	Implements a Softmax Unit.
	"""
	def __init__(self):
		super(Softmax, self).__init__(hasParams = False)
		self.name = "NNUnit.Softmax"

	def forward(self, input):
		cappedLL = input - np.max(input, axis=1, keepdims=True)
		# To prevent softmax explosions
		expLayer = np.power(np.e, cappedLL)
		output = expLayer / np.sum(expLayer, axis=1)[:, np.newaxis]
		self.lastInput = input
		self.lastOutput = output
		return self.forwardprop()

	def d_input(self, grads):
		"""
		The gradient for softmax is computed as follows: 

		dLogit_i/dInput_j = - SM[i] * SM[j] 		(for i != j)
	 	dLogit_i/dInput_i = - SM[i] * SM[i] + SM[i]  

	 	"""
		out = self.lastOutput
		s0, s1 = out.shape
		out1 = out.reshape((s0, s1, 1))
		out2 = -out.reshape((s0, 1, s1))
 
		jacobian = np.matmul(out1, out2)

		# Multiplies (nBatch x [I_{nClasses}]) with (nBatch x nClasses) = (nBatch x nClasses x nClasses)
		diagonal = np.eye(s1)
		diagonal = np.repeat(diagonal[np.newaxis,:,:], s0, axis=0)
		diagonal = diagonal * out1
		
		jacobian += diagonal

		# Multiplies a (nBatch x 1 x nClasses) with (nBatch, nClasses, nClasses)
		newGrad = np.matmul(grads.reshape((s0, 1, s1)), jacobian).reshape((s0, s1))
		return newGrad


class CrossEntropyLoss():
	"""
	Implements Cross Entropy Loss.
	"""
	def __init__(self):
		self.zero_grad()

	def print_loss(self, pred, label, loss):
		if DEBUG_MODE:
			print "Predicted", pred
			print "Label", label

	def eval(self, pred, label):
		losses = label * np.log(pred)
		loss = -np.sum(losses, axis=1).reshape((-1,1))
		self.gradient = self.d_pred(loss, label, pred)
		self.print_loss(pred, label, loss)
		return loss

	def d_pred(self, loss, label, pred):
		return -label * (1.0/pred) * loss

	def zero_grad(self):
		self.gradient = 0




