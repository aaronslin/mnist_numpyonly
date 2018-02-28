import numpy as np
import time
import nn

MNIST_FLAT = (784,)
N_CLASSES = (10,)

def read_CSV(file_obj):
	while True:
		data = file_obj.readline()
		if not data:
			assert "No more data."
			break
		yield data

def clean_data(line):
	data = line.split(",")

	# Processing the label into a one-hot vector
	labelNum = int(data.pop(0))
	label = np.zeros(N_CLASSES)
	label[labelNum] = 1

	# Processing the image into a float
	image = np.array(data).astype(np.float32)
	return image, label

def get_batch():
	pass
	# Might be good to put this in readCSV

class FeedForward():
	def __init__(self, fc1, fc2):
		self.inp_size = 784
		self.out_size = 10
		self.fc1_size = fc1
		self.fc2_size = fc2

		self.linear1 = nn.LinearUnit(self.inp_size, self.fc1_size)
		self.linear2 = nn.LinearUnit(self.fc1_size, self.fc2_size)
		self.linear3 = nn.LinearUnit(self.fc2_size, self.out_size)

		# Manual hack to get the set of trainable units
		self.trainable = [self.linear1, self.linear2, self.linear3]
		self.architecture = [self.linear1, nn.Relu, self.linear2, nn.Relu, self.linear3, nn.Softmax]

	def forward(self, input):
		lastComputation = input
		for unit in self.architecture:
			lastComputation = unit.forward(lastComputation)
		return lastComputation

	def backward(self, loss, input):
		# Inefficiency: propagates backwards once for each unit in self.trainable
		# 				instead of propagating backwards once
		gradients = []
		archBack = list(reversed(self.architecture))
		archBack.append(None)
		prevInput = [ for unit in archBack



		for paramUnit in self.trainable:
			error = loss
			for unit in archBack:
				if paramUnit == unit:
					grad = paramUnit.d_weights(error, inputFromPrevLayer)
					gradients.append(grad)
				else:
					error = unit.d_input(error, inputFromPrevLayer)


"""
Where I left off (9:30):

 - Chaining gradients together from nn.py modules
 	Difficulty: the modules need to know the inputs from previous layers 
 	in order to backprop
 	Having issues with __super__
 - Absolutely need to write test cases

"""


# class NN_old():
# 	def __init__(self, fc1, fc2):

# 		self.initialize_weights()
# 		self.loss = self.xentropy

# 	def initialize_weights(self):
# 		self.W1 = np.random.normal(size=(self.inp_size, self.fc1_size))
# 		self.W2 = np.random.normal(size=(self.fc1_size, self.fc2_size))
# 		self.W3 = np.random.normal(size=(self.fc2_size, self.out_size))

# 	def forward(self, input):
# 		"""
# 		input has size batch_size X inp_size
# 		output has size batch_size X out_size
# 		"""
# 		fc1 = np.matmul(input, self.W1)
# 		fc1 = self.relu(fc1)

# 		fc2 = np.matmul(fc1, self.W2)
# 		fc2 = self.relu(fc2)

# 		fc3 = np.matmul(fc2, self.W3)
# 		output = self.softmax(fc3)

# 		return output

# 	def backward(self, label, pred):
# 		loss = self.loss(label, pred)
# 		pass

# 	def relu(self, layer):
# 		pass

# 	def softmax(self, layer):
# 		pass

# 	def xentropy(self, label, pred):
# 		pass





	



			

if __name__ == "__main__":
	filename="mnist_train.csv"
	with open(filename) as file:
		for line in read_CSV(file):
			clean_data(line)

			break

