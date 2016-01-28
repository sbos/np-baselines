import theano as th
import lasagne
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from prepare_data import *

input_img = T.matrix('X')
D = 20 * 20

code_length = 64


class SaltAndPepperNoiseLayer(lasagne.layers.Layer):
    def __init__(self, incoming, rate=0.1, **kwargs):
        super(SaltAndPepperNoiseLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(np.random.randint(1, 2147462579))
        self.rate = rate

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.rate == 0:
            return input
        else:
            drop = self._srng.uniform(input.shape)
            z = T.lt(drop, 0.5 * self.rate)
            o = T.lt(T.abs_(drop - 0.75 * self.rate), 0.25 * self.rate)
            input = T.set_subtensor(input[z.nonzero()], 0.)
            input = T.set_subtensor(input[o.nonzero()], 1.)
            return input

from lasagne.nonlinearities import leaky_rectify, sigmoid
encoder = lasagne.layers.InputLayer((None, D), input_img)
#encoder = SaltAndPepperNoiseLayer(encoder)
encoder = lasagne.layers.DenseLayer(encoder, 200, nonlinearity=leaky_rectify)
encoder = lasagne.layers.DenseLayer(encoder, 200, nonlinearity=leaky_rectify)
code = lasagne.layers.DenseLayer(encoder, code_length, nonlinearity=leaky_rectify)

get_code = th.function([input_img], lasagne.layers.get_output(code, deterministic=True))

decoder = lasagne.layers.DenseLayer(code, 200, nonlinearity=leaky_rectify)
decoder = lasagne.layers.DenseLayer(code, 200, nonlinearity=leaky_rectify)
decoder = lasagne.layers.DenseLayer(code, D, nonlinearity=sigmoid)

reconstruction = lasagne.layers.get_output(decoder)
loss = lasagne.objectives.binary_crossentropy(reconstruction, input_img).mean()
loss += 1e-8 * lasagne.regularization.regularize_network_params(decoder, lasagne.regularization.l2)

params = lasagne.layers.get_all_params(decoder, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=1e-4)

train_fn = th.function([input_img], loss, updates=updates)
evaluate_loss = th.function([input_img], loss)
get_rec = th.function([input_img], reconstruction)

