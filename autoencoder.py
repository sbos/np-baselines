import theano as th
import lasagne
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

input_img = T.tensor2('X')
D = 28 * 28

code_length = 200


class SaltAndPepperNoiseLayer(lasagne.layers.Layer):
    def __init__(self, incoming, sigma=0.1, **kwargs):
        super(SaltAndPepperNoiseLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(np.random.randint(1, 2147462579))
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.sigma == 0:
            return input
        else:
            # fixme
            return input + T.abs_(self._srng.normal(input.shape,
                                             avg=0.0,
                                             std=self.sigma))

from lasagne.nonlinearities import leaky_rectify, sigmoid
encoder = lasagne.layers.InputLayer((None, D), input_img)
encoder = lasagne.layers.GaussianNoiseLayer(encoder)
encoder = lasagne.layers.DenseLayer(encoder, 200, nonlinearity=leaky_rectify)
encoder = lasagne.layers.DenseLayer(encoder, 200, nonlinearity=leaky_rectify)
code = lasagne.layers.DenseLayer(encoder, code_length, nonlinearity=leaky_rectify)

decoder = lasagne.layers.DenseLayer(code, 200, nonlinearity=leaky_rectify)
decoder = lasagne.layers.DenseLayer(code, 200, nonlinearity=sigmoid)

reconstruction = lasagne.layers.get_output(decoder)
loss = lasagne.objectives.binary_crossentropy(reconstruction, input_img).mean()
loss += 1e-6 * lasagne.regularization.regularize_network_params(decoder, lasagne.regularization.l2)

params = lasagne.layers.get_all_params(decoder, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=1e-4)

train_fn = th.function([input_img], loss, updates=updates)

