import nn
from graph import FeedForward
import numpy as np

inp_size = 2
fc1_size = 3
fc2_size = 6
out_size = 10

arch = [nn.Linear, nn.Relu, nn.Linear, nn.Relu, nn.Linear, nn.Softmax]
dims = [(inp_size, fc1_size), None, (fc1_size, fc2_size), None, (fc2_size, out_size), None]
lossF = nn.CrossEntropyLoss
learning_rate = 0.05

def ____hrule():
	print "\n **********************************************************\n"

def test1():
	"""
	Not exactly a unit test. A function written to step through and debug
	forward and backward passes.
	"""
	ff = FeedForward(arch, dims, lossF)
	ff.initialize_weights("uniform")

	input = np.zeros((1, inp_size))
	input = np.arange(inp_size).reshape((1, -1))
	label = np.array([1,0,0,0,0,0,0,0,0,0])

	output = ff.forward(input)
	____hrule()

	loss = ff.eval(output, label)
	print loss
	print "Label:", label
	print "Predicted:", output
	____hrule()


	ff.backward(loss, learning_rate)
	____hrule()
	print "New Weights"
	for layer in ff.layers:
		if layer.hasParams:
			print layer.name, layer.weights

	ff.clear_history()

def test_dSoftmax():
	"""
	Unit test for nn.Softmax().d_input
	"""
	sm = nn.Softmax()
	output = 0.1*np.array([0, 1, 2, 1, 2, 3]).reshape((2,3))
	sm.lastOutput = output
	grads = np.ones(6).reshape((2,3))
	newGrad = sm.d_input(grads)


	"""
	(Computed by hand)

	Expected Jacobian: (2x3x3)
		[ [0, 0, 0], [0, 0.09, -0.02], [0, -0.02, 0.16] ]
		[ [0.09, -0.02, -0.03], [-0.02, 0.16, -0.06], [-0.03, -0.06, 0.21]]

	Expected newGrad: (2x3)
		[ [0, 0.07, 0.14], 
		  [0.04, 0.08, 0.12] ]
	"""



if __name__ == "__main__":
	test1()
	test_dSoftmax()




