
class NNUnit():
	def __init__(self):
		self.parentUnit = None
		self.childUnit = None


class LinearUnit(NNUnit):
	def __init__(self, inp_size, out_size, bias=None):
		super(LinearUnit, self).__init__()

		self.inp_size = inp_size
		self.out_size = out_size
		self.weights = initialize_weights()

		self.lastInput = None
		self.lastOutput = None

	def initialize_weights(self):
		return np.random.normal(size=(self.inp_size, self.out_size))

	def forward(self, input):
		output = np.matmul(input, self.weights)
		self.lastInput = input
		self.lastOutput = output
		return output

	def d_weights(self, loss, input):
		return np.matmul(input.T, loss)

	def d_input(self, loss, input):
		return np.matmul(loss, self.weights.T)

class Relu(NNUnit):
	def __init__(self, input):
		super(Relu, self).__init__()
		self.forward(input)

	def forward(self, input):
		input[input < 0] = 0
		return input

	def d_input(self, loss, input):
		return loss * input[input > 0]

class Softmax(NNUnit):
	def __init__(self, input):
		super(Softmax, self).__init__()
		self.forward(input)

	def forward(self, input):
		expLayer = np.power(np.e, input)
		return expLayer / np.sum(expLayer)

	def d_input(self, loss, input):
		"""
		Loss: batch X nClasses
		Input: batch X nClasses
		Uses the property that dSoftmax(x)/dx = Softmax(x)(1-Softmax(x))
		"""
		out = self.forward(input)
		return loss * (out*(1-out))

class CrossEntropyLoss():
	def __init__(self, label, pred):
		self.forward(label, pred)

	def forward(self, label, pred):
		"""
		label is one-hot, should be batch_size X out_size
		pred should be batch_size X out_size
		output is batch_size X ,
		"""
		losses = label * np.log(pred)
		return np.sum(losses, axis=1)

	def d_pred(self, loss, label, pred):
		return -label * (1.0/pred) * loss

if __name__ == "__main__":
	X = [LinearUnit(1, 2), Relu, LinearUnit(2, 3), Relu]
	Y = [x.forward(0) for x in X]
	print Y




