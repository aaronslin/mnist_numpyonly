# tests.py

import nn
from graph import FeedForward
import numpy as np

def ____hrule():
	print "\n **********************************************************\n"

def test1():
	inp_size = 2
	fc1_size = 3
	fc2_size = 6
	out_size = 10

	arch = [nn.Linear, nn.Relu, nn.Linear, nn.Relu, nn.Linear, nn.Softmax]
	dims = [(inp_size, fc1_size), None, (fc1_size, fc2_size), None, (fc2_size, out_size), None]
	lossF = nn.CrossEntropyLoss
	learning_rate = 0.05
	ff = FeedForward(arch, dims, lossF, learning_rate)
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


	ff.backward(loss)
	____hrule()
	print "New Weights"
	for layer in ff.layers:
		if layer.hasParams:
			print layer.name, layer.weights


test1()