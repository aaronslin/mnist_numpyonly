import numpy as np

N_CLASSES = 10

def csv_generator(file_obj):
	"""
	Given a file object, returns a generator to iterate through training 
	data and labels. A generator was chosen over a list to minimize buffer 
	usage and overhead startup cost
	"""
	while True:
		data = file_obj.readline()
		if not data:
			assert "No more data."
			break
		yield data

def clean_data(data):
	data = data.split(",")

	# Processing the label into a one-hot vector
	labelNum = int(data.pop(0))
	label = np.zeros(N_CLASSES)
	label[labelNum] = 1

	# Processing the image into a float, scaling pixels down
	# Implements mean-only batch normalization per image
	image = np.array(data).astype(np.float32)
	image = (image - np.mean(image))/np.sum(image)
	return image, label


def get_batch(batch_size, generator, filename):
	"""
	Given a batch_size and raw data generator, produces batches of clean
	labels and images to use for training/test.

	If the generator is emptied, then a new generator of training images 
	needs to be created to maintain a constant batch size. Because of these
	cases, get_batch() requires filename as input and returns generator. 

	Admittedly, this process can be implemented more elegantly without
	requiring filename as an input. 

	One possible extension is to randomize batches. Currently, images are
	loaded in a determistic order.
	"""
	images = []
	labels = []
	for i in range(batch_size):
		data = next(generator, None)
		# If the generator runs out of data, then data = None
		if data is None:
			generator = csv_generator(open(filename))
			data = next(generator)
		img, lbl = clean_data(data)
		images.append(img)
		labels.append(lbl)
	images = [img for img in images if img is not None]
	labels = [lbl for lbl in labels if lbl is not None]
	return generator, np.array(images), np.array(labels)