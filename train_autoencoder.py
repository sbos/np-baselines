from autoencoder import *
from prepare_data import *
from params import *
import gc

batch_size = 200

all_data = dict()
parse_dataset('./all', all_data)

train, test = split_dataset(all_data, 1200)

# for class_name, images in test.iteritems():
#     train[class_name] = images[:10]

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
    if j == 417:
        write_model_data(decoder, 'checkpoint417')
    print time.clock() - start
