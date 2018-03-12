# mnist_numpyonly

This project is a challenge to train a deep neural network to classify MNIST digits, using only numpy as a dependency.

### Running the script

The only dependency is numpy. The datasets were obtained from [1].

To run the script, enter:

```python main.py```

With the preset parameters, the network trains for <60 seconds (1000 epochs) and obtains a test accuracy of ~51%. With 10000 epochs, the network obtains a test accuracy of 77%.

### Modules

* ```main.py```: The main script to run.
* ```data.py```: Loads and processes training and test data
* ```nn.py```: Implements NNUnit, the basic class for neural network layers, and instances of this. Additionally, implements a loss function.
* ```graph.py```: Class for computation graphs. Only supports straight, feedforward networks currently.
* ```train.py```: Modules for training, testing, and scoring a model.
* ```test.py```: For testing of implemented modules

### Thoughts

Here are the main design choices I made with some additional thoughts:

* I wanted to write my code in a way that can easily be expanded to include different types of computations and architectures that are not currently specified in this toy problem. For example:
  * NNUnit is a very general class to represent a layer of a neural network. It allows the instantiation of additional layers, both with trainable parameters (e.g. a convolutional layer) and without trainable parameters (e.g. pooling layers).
  * FeedForward is also a general class that currently only supports the simple feedforward model for a neural network. However, it (together with NNUnit) can easily be extended to support arbitrary DAG structured graphs. Each NNUnit would keep a list of all parents and children. On a forward pass, the DAG can be topologically sorted to give a valid computation order. On the backwards pass, we accumulate gradients from each unit's children and sum them up.
* Computations were written to be vectorized wherever possible. This greatly speeds up computations.
* A big part of the challenge was properly designating various properties of the layer units and computation graphs. Originally, I intended to run a separate backwards pass for each trainable layer in the architecture, for ease and simplicity. For the MNIST network (with 3 linear units), this doesn't hinder the runtime significantly, but ultimately I decided to introduce the parent and child properties of layers, rather than manually propagate inputs or gradients through the network.
* SGD is currently intrisic to FeedForward networks. This assumption greatly increases the modularity of the remaining code for this MNIST classifier, but should another optimizer be needed, there isn't a general framework for doing so.
* One extension is to include bias terms in the Linear units.



[1] https://pjreddie.com/projects/mnist-in-csv/
