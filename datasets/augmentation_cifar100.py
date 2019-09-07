import datasets.augmentation_transforms as augmentation_transforms
import numpy as np
import datasets.policies as found_policies
import chainer


class Dataset(chainer.dataset.DatasetMixin):

    def __init__(self):
        train, test = chainer.datasets.get_cifar100(dtype=np.float64)
        images, labels = chainer.dataset.concat_examples(train)
        images = images.transpose(0, 2, 3, 1).copy()
        mean = augmentation_transforms.MEANS
        std = augmentation_transforms.STDS

        images = (images-mean)/std
        self.train = chainer.datasets.TupleDataset(images, labels)
        self.good_policies = found_policies.good_policies()

    def __len__(self):
        return len(self.train)

    def get_example(self, i):
        policy = self.good_policies[np.random.choice(len(self.good_policies))]
        image, label = self.train[i]
        image = augmentation_transforms.apply_policy(policy, image)
        image = augmentation_transforms.zero_pad_and_crop(image, 4)
        image = augmentation_transforms.random_flip(image)
        image = augmentation_transforms.cutout_numpy(image)
        image = image.transpose(2, 0, 1).astype(np.float32)
        return image, label


class DatasetWithoutLabel(chainer.dataset.DatasetMixin):

    def __init__(self):
        train, test = chainer.datasets.get_cifar10(dtype=np.float64)
        images, labels = chainer.dataset.concat_examples(train)
        images = images.transpose(0, 2, 3, 1).copy()
        mean = augmentation_transforms.MEANS
        std = augmentation_transforms.STDS

        images = (images-mean)/std
        self.train = chainer.datasets.TupleDataset(images, labels)
        self.good_policies = found_policies.good_policies()

    def __len__(self):
        return len(self.train)

    def get_example(self, i):
        policy = self.good_policies[np.random.choice(len(self.good_policies))]
        image, label = self.train[i]
        image = augmentation_transforms.apply_policy(policy, image)
        image = augmentation_transforms.zero_pad_and_crop(image, 4)
        image = augmentation_transforms.random_flip(image)
        image = augmentation_transforms.cutout_numpy(image)
        image = image.transpose(2, 0, 1).astype(np.float32)
        return image


class TestDataset(chainer.dataset.DatasetMixin):

    def __init__(self):
        train, test = chainer.datasets.get_cifar100(dtype=np.float64)
        images, labels = chainer.dataset.concat_examples(test)
        images = images.transpose(0, 2, 3, 1).copy()
        mean = augmentation_transforms.MEANS
        std = augmentation_transforms.STDS

        images = (images-mean)/std
        self.test = chainer.datasets.TupleDataset(images, labels)

    def __len__(self):
        return len(self.test)

    def get_example(self, i):
        image, label = self.test[i]
        image = image.transpose(2, 0, 1).astype(np.float32)
        return image, label
