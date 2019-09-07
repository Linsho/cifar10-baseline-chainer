import numpy as np
import chainer


MEANS = [0.49139968, 0.48215841, 0.44653091]
STDS = [0.24703223, 0.24348513, 0.26158784]


class TrainDataset(chainer.dataset.DatasetMixin):

    def __init__(self):
        train, test = chainer.datasets.get_cifar10(dtype=np.float64)
        images, labels = chainer.dataset.concat_examples(train)
        images = images.transpose(0, 2, 3, 1).copy()
        mean = MEANS
        std = STDS

        images = (images-mean)/std
        self.train = chainer.datasets.TupleDataset(images, labels)

    def __len__(self):
        return len(self.train)

    def get_example(self, i):
        image, label = self.train[i]
        image = image.transpose(2, 0, 1).astype(np.float32)
        return image, label
    

class TestDataset(chainer.dataset.DatasetMixin):

    def __init__(self):
        train, test = chainer.datasets.get_cifar10(dtype=np.float64)
        images, labels = chainer.dataset.concat_examples(test)
        images = images.transpose(0, 2, 3, 1).copy()
        mean = MEANS
        std = STDS

        images = (images-mean)/std
        self.test = chainer.datasets.TupleDataset(images, labels)

    def __len__(self):
        return len(self.test)

    def get_example(self, i):
        image, label = self.test[i]
        image = image.transpose(2, 0, 1).astype(np.float32)
        return image, label