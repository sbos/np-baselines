import numpy as np
import os
import path
from PIL import Image
import matplotlib.pyplot as plt


def parse_dataset(dir, x):
    basedir = dir
    subdirlist = []
    for item in os.listdir(dir):
        filename = os.path.join(basedir, item)
        if os.path.isfile(filename):
            if basedir not in x:
                x[basedir] = []
            x[basedir].append(filename)
        else:
            subdirlist.append(filename)
    for subdir in subdirlist:
        parse_dataset(subdir, x)


def split_dataset(data, num_train):
    count = 0
    train = dict()
    test = dict()

    for class_name, images in data.iteritems():
        if count < num_train:
            train[class_name] = images
        else:
            test[class_name] = images

        count += 1

    return train, test


def safe_rotate(img, angle):
    fff = Image.new('RGBA', img.size, 'white')
    rot = img.rotate(angle, expand=1).resize(img.size)
    return Image.composite(rot, fff, rot).convert(img.mode)


def augment_data(files):
    x = []
    class_rotation = np.random.choice([0, 90, 180, 270])
    for img in files:
        dx = np.random.randint(0, 10)
        dy = np.random.randint(0, 10)

        translated = Image.new('RGBA', img.size, 'white')
        translated.paste(img, (dx, dy))
        translated = safe_rotate(translated, np.random.uniform(-90./16, 90/16.))
        translated = translated.resize((20, 20))
        translated = safe_rotate(translated, class_rotation)
        x.append(np.asarray(translated.getdata(0), dtype=np.float32).ravel())
    return np.vstack(x)


def load_dataset(data):
    new_data = dict()

    for class_name, files in data.iteritems():
        x = []
        for fname in files:
            img = Image.open(fname)
            img.load()
            x.append(img)
            #x.append(img.copy())
            #img.close()
        new_data[class_name] = x

    return new_data


def augment_dataset(data, count=0):
    new_data = dict()

    for class_name, files in data.iteritems():
        new_data[class_name] = augment_data(files)

    return new_data

