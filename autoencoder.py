import theano as th
import lasagne
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from prepare_data import *
import gc
from params import *

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
# encoder = SaltAndPepperNoiseLayer(encoder)
encoder = lasagne.layers.DenseLayer(encoder, 400, nonlinearity=leaky_rectify)
encoder = lasagne.layers.DenseLayer(encoder, 400, nonlinearity=leaky_rectify)
code = lasagne.layers.DenseLayer(encoder, code_length, nonlinearity=leaky_rectify)

decoder = lasagne.layers.DenseLayer(code, 400, nonlinearity=leaky_rectify)
decoder = lasagne.layers.DenseLayer(code, D, nonlinearity=sigmoid)

reconstruction = lasagne.layers.get_output(decoder)
loss = lasagne.objectives.binary_crossentropy(reconstruction, input_img).mean()
# loss += 1e-6 * lasagne.regularization.regularize_network_params(decoder, lasagne.regularization.l2)

params = lasagne.layers.get_all_params(decoder, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=1e-4)

train_fn = th.function([input_img], loss, updates=updates)
evaluate_loss = th.function([input_img], loss)
get_rec = th.function([input_img], reconstruction)

batch_size = 200

all_data = dict()
parse_dataset('./all', all_data)

train, test = split_dataset(all_data, 1200)
train = load_dataset(train)

import time

if os.path.exists('checkpoint.params'):
    print 'checkpoint exists, continuing'
    read_model_data(decoder, 'checkpoint')

for j in xrange(3 * 417):
    start = time.clock()
    train_data = augment_dataset(train)

    train_data = np.vstack(train_data.values())
    N = train_data.shape[0]
    for i in xrange(N / batch_size):
        batch = np.random.choice(N, batch_size)
        print i, j, train_fn(train_data[batch] / 255.)
        if i % 50 == 0:
            test_data = train_data[np.random.choice(N, 5)] / 255.
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.matshow(get_rec(test_data).reshape(5 * 20, 20))
            ax2.matshow(test_data.reshape(5 * 20, 20))
            f.savefig('rec.png')
            f.clf()
            plt.clf()
            plt.close()
            del f
    gc.collect()
    write_model_data(decoder, 'checkpoint')
    print time.clock() - start
