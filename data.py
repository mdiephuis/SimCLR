from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np

from PIL import ImageFilter
from PIL import Image


class GaussianSmoothing(object):
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(self.radius))


def cifar_train_transforms():
    all_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    return all_transforms


def cifar_test_transforms():
    all_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    return all_transforms


class CIFAR10C(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CIFAR10C, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            xi = self.transform(img)
            xj = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return xi, xj, target


class Loader(object):
    def __init__(self, dataset_ident, file_path, download, batch_size, train_transform, test_transform, target_transform, use_cuda):

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

        loader_map = {
            'CIFAR10C': CIFAR10C,
            'CIFAR10': datasets.CIFAR10
        }

        num_class = {
            'CIFAR10C': 10,
            'CIFAR10': 10
        }

        # Get the datasets
        train_dataset, test_dataset = self.get_dataset(loader_map[dataset_ident], file_path, download,
                                                       train_transform, test_transform, target_transform)
        # Set the loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        tmp_batch = self.train_loader.__iter__().__next__()[0]
        self.img_shape = list(tmp_batch.size())[1:]
        self.num_class = num_class[dataset_ident]

    @staticmethod
    def get_dataset(dataset, file_path, download, train_transform, test_transform, target_transform):

        # Training and Validation datasets
        train_dataset = dataset(file_path, train=True, download=download,
                                transform=train_transform,
                                target_transform=target_transform)

        test_dataset = dataset(file_path, train=False, download=download,
                               transform=test_transform,
                               target_transform=target_transform)

        return train_dataset, test_dataset
