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



"""
Where I left off (9:30):

 - Chaining gradients together from nn.py modules
 	Difficulty: the modules need to know the inputs from previous layers 
 	in order to backprop
 	Having issues with __super__
 - Absolutely need to write test cases

"""





	



			

if __name__ == "__main__":
	filename="mnist_train.csv"
	with open(filename) as file:
		for line in read_CSV(file):
			clean_data(line)

			break

