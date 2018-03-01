import numpy as np
import nn
from graph import FeedForward
from train import train, test


# Constants
MNIST_SIZE = 784
N_CLASSES = 10
MAX_PIXEL = 256.0

# Data Preparation
trainFile = "mnist_train.csv"
testFile = "mnist_test.csv"
			
# Architecture-specific Parameters
inp_size = MNIST_SIZE
fc1_size = 128
fc2_size = 64
out_size = N_CLASSES


if __name__ == "__main__":
	arch = [nn.Linear, nn.Relu, nn.Linear, nn.Relu, nn.Linear, nn.Softmax]
	dims = [(inp_size, fc1_size), None, (fc1_size, fc2_size), None, (fc2_size, out_size), None]
	lossFunc = nn.CrossEntropyLoss

	# Model is created
	ff = FeedForward(arch, dims, lossFunc)
	ff.initialize_weights("random")

	# Test and train
	train(ff, trainFile)
	test(ff, testFile)






