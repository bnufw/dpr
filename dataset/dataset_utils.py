import os
import sys
from typing import Any, Tuple

import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

def find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def get_train_val_images_tiny(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    train_dir = os.path.join(data_path, 'tiny-imagenet-200', 'train')
    test_dir = os.path.join(data_path, 'tiny-imagenet-200', 'val')
    train_dset = datasets.ImageFolder(train_dir)

    train_images = []
    train_labels = []
    for item in train_dset.imgs:
        train_images.append(item[0])
        train_labels.append(item[1])

    train_targets =  np.array(train_labels)

    test_images = []
    test_labels = []
    _, class_to_idx = find_classes(train_dir)
    imgs_path = os.path.join(test_dir, 'images')
    imgs_annotations = os.path.join(test_dir, 'val_annotations.txt')
    with open(imgs_annotations) as r:
        data_info = map(lambda s: s.split('\t'), r.readlines())
    cls_map = {line_data[0]: line_data[1] for line_data in data_info}
    for imgname in sorted(os.listdir(imgs_path)):
        if cls_map[imgname] in sorted(class_to_idx.keys()):
            path = os.path.join(imgs_path, imgname)
            test_images.append(path)
            test_labels.append(class_to_idx[cls_map[imgname]])

    test_targets =  np.array(test_labels)

    return train_images, train_targets, test_images, test_targets, class_to_idx

def get_train_val_images_imagenet_subset(data_path):
    train_dir = os.path.join(data_path, 'imagenet_subset', 'train')
    train_dset = datasets.ImageFolder(train_dir)
    split_dir = os.path.join(os.path.dirname(__file__), 'splits')
    with open(os.path.join(split_dir, 'imagenet_subset_classes.txt')) as source:
        classes = source.read().splitlines()
    if train_dset.classes != classes:
        raise ValueError(
            'ImageNet-Subset train must contain exactly the 100 class directories '
            'listed in dataset/splits/imagenet_subset_classes.txt')

    train_images = []
    train_labels = []
    for item in train_dset.imgs:
        train_images.append(item[0])
        train_labels.append(item[1])

    train_targets =  np.array(train_labels)
    class_to_idx = train_dset.class_to_idx

    test_images = []
    test_labels = []
    with open(os.path.join(split_dir, 'imagenet_subset_val.txt')) as source:
        for line in source:
            image_path, label = line.split()
            test_images.append(os.path.join(data_path, 'imagenet_subset', image_path))
            test_labels.append(int(label))

    test_targets =  np.array(test_labels)

    return train_images, train_targets, test_images, test_targets, class_to_idx

class TinyImagenetDataset(Dataset):
    def __init__(self, data, targets, class_to_idx,  transform):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img_path, target = self.data[index], self.targets[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

def get_dataset(dataset_type, data_path, n_task):
    if  dataset_type == "cifar100":
        print("Loading Cifar 100")
        train_transform = [transforms.Pad(4), transforms.RandomResizedCrop(32),
                           transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                           transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
        train_transform = transforms.Compose(train_transform)

        test_transform = [transforms.ToTensor(),transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
        test_transform = transforms.Compose(test_transform)

        if dataset_type == "cifar100":
            train_set = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                download=False, transform=train_transform)
            test_set = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                            download=False, transform=test_transform)
            n_classes = 100

    elif dataset_type == "tiny-imagenet":
        # images_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        print("Loading Tiny Imagenet")

        train_data, train_targets, test_data, test_targets, class_to_idx = get_train_val_images_tiny(data_path)

        train_transform = transforms.Compose(
                                            [transforms.RandomCrop(64, padding=8),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                            ])

        test_transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

        train_set = TinyImagenetDataset(train_data, train_targets, class_to_idx, train_transform)
        test_set = TinyImagenetDataset(test_data, test_targets, class_to_idx, test_transform)

        n_classes = 200

    elif dataset_type == "imagenet-subset":
        print("Loading {}".format(dataset_type))

        train_data, train_targets, test_data, test_targets, class_to_idx = get_train_val_images_imagenet_subset(data_path)

        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                    ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])

        train_set = TinyImagenetDataset(train_data, train_targets, class_to_idx, train_transform)
        test_set = TinyImagenetDataset(test_data, test_targets, class_to_idx.copy(), test_transform)

        n_classes = 100
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_type))
    return train_set, test_set, n_classes
