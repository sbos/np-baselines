from autoencoder import *
from params import *
import numpy as np

read_model_data(decoder, 'checkpoint')

all_data = dict()
parse_dataset('./all', all_data)

train, test = split_dataset(all_data, 1200)
test = load_dataset(test)
test = augment_dataset(test)

K = 1
num_classes = 5


def test_episode(x, y, stats):
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    sim = np.dot(x, x.T)
    votes = np.zeros(num_classes)
    for i in xrange(len(y)):
        xsim = sim[i, :i]
        nearest = xsim.argsort()[-K:]

        votes[:] = 0.
        for j in nearest:
            votes[y[j]] += xsim[j]
        y_hat = votes.argmax()

        if y_hat == y[i]:
            stats[np.sum(y[:i+1] == y[i])] += 1

num_episodes = 10000
stats = np.zeros(1000)

for episode in xrange(num_episodes):
    classes = np.random.choice(test.keys(), num_classes)
    x = []
    x_features = []
    y = []

    for k in xrange(num_classes):
        class_data = test[classes[k]]
        x.append(class_data)
        x_features.append(get_code(class_data))
        y.append(np.ones(class_data.shape[0], dtype=np.int32) * k)

    x_raw = np.vstack(x)
    x_features = np.vstack(x_features)
    y = np.hstack(y)

    perm = np.random.permutation(len(y))

    x_raw = x_raw[perm]
    x_features = x_features[perm]
    y = y[perm]

    test_episode(x_features, y, stats)

avg_stats = stats / num_episodes / num_classes

print avg_stats[1], avg_stats[2], avg_stats[3], avg_stats[4], avg_stats[5], avg_stats[10]
