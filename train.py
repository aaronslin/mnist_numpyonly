import numpy as np
from data import get_batch, csv_generator

def train(model, trainFile):
	n_epochs = 1000
	batch_size = 100
	print_every = 5
	learnRate = .0001
	trainGen = csv_generator(open(trainFile))

	for epoch in range(n_epochs):
		trainGen, x_batch, y_batch = get_batch(batch_size, trainGen, trainFile)

		predicted = model.forward(x_batch)
		loss = model.eval(predicted, y_batch)
		model.backward(loss, learnRate)

		if epoch % print_every == 0:
			print "Epoch: ", epoch, "\tTrainAccuracy: ", score_accuracy(predicted, y_batch)

		model.clear_history()
	print "Training complete!"

def test(model, testFile):
	batch_size = 1000
	testGen = csv_generator(open(testFile))

	testGen, x_batch, y_batch = get_batch(batch_size, testGen, testFile)
	predicted = model.forward(x_batch)
	accuracy = score_accuracy(predicted, y_batch)
	print "Test Accuracy (batch_size=", batch_size, "):", accuracy

def score_accuracy(predicted, labels):
	"""
	Returns percent of rows in predicted and labels that have
	the same max. (i.e. the correct prediction)
	"""
	batch_size = predicted.shape[0]
	maxCategory = np.zeros_like(predicted)
	maxCategory[np.arange(batch_size), np.argmax(predicted, axis=1)] = 1
	accuracy = 1.0 * np.sum(maxCategory * labels)/batch_size
	return accuracy	