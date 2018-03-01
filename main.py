import numpy as np
import time
import nn
from graph import FeedForward

MNIST_SIZE = 784
N_CLASSES = 10
MAX_PIXEL = 256.

def csv_generator(file_obj):
	while True:
		data = file_obj.readline()
		if not data:
			assert "No more data."
			break
		yield data

def clean_data(data):
	if data is None:
		return None, None
	data = data.split(",")

	# Processing the label into a one-hot vector
	labelNum = int(data.pop(0))
	label = _one_hot(labelNum, N_CLASSES)

	# Processing the image into a float
	image = np.array(data).astype(np.float32)
	image = (image - np.mean(image))/np.sum(image)
	return image, label

def _one_hot(digit, total):
	v = np.zeros(total)
	v[digit] = 1
	return v


def get_batch(batch_size, generator, filename):
	images = []
	labels = []
	for i in range(batch_size):
		data = next(generator, None)
		if data is None:
			generator = csv_generator(open(filename))
			print "Generator depleted. Training data recycling."
			data = next(generator)
		# If the generator runs out of data, then data = None
		img, lbl = clean_data(data)
		images.append(img)
		labels.append(lbl)
	images = [img for img in images if img is not None]
	labels = [lbl for lbl in labels if lbl is not None]
	return generator, np.array(images), np.array(labels)

def score_accuracy(predicted, labels):
	batch_size = predicted.shape[0]
	maxCategory = np.zeros_like(predicted)
	maxCategory[np.arange(batch_size), np.argmax(predicted, axis=1)] = 1
	accuracy = 1.0 * np.sum(maxCategory * labels)/batch_size
	return accuracy	



# Parameters
n_epochs = 10000
batch_size = 100
learnRate = .0001
print_every = 5

# Data preparation
trainFile = "mnist_train.csv"
trainGen = csv_generator(open(trainFile))
			
# Architecture specific
inp_size = MNIST_SIZE
fc1_size = 128
fc2_size = 64
out_size = N_CLASSES

arch = [nn.Linear, nn.Relu, nn.Linear, nn.Relu, nn.Linear, nn.Softmax]
dims = [(inp_size, fc1_size), None, (fc1_size, fc2_size), None, (fc2_size, out_size), None]
lossFunc = nn.CrossEntropyLoss

# FF is created
ff = FeedForward(arch, dims, lossFunc, learnRate)
ff.initialize_weights("random")

# Training
for epoch in range(n_epochs):
	trainGen, x_batch, y_batch = get_batch(batch_size, trainGen, trainFile)

	predicted = ff.forward(x_batch)
	loss = ff.eval(predicted, y_batch)
	ff.backward(loss)

	if epoch % print_every == 0:
		print "Epoch: ", epoch, "\tAccuracy: ", score_accuracy(predicted, y_batch)

		weightsums = [np.sum(ff.layers[i].weights) for i in [0,2,4]]
		#print "\t", weightsums[0], "\t", weightsums[1], "\t",weightsums[2]
		if sum(weightsums) != sum(weightsums):
			break
	ff.clear_history()



if __name__ == "__main__":
	pass






