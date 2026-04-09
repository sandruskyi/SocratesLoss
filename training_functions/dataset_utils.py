"""
Extended by
@User: sandruskyi
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms, utils, datasets
from PIL import Image
import json


__all__ = [ 'C10', 'SVHN', 'ImageNetKaggle', 'MNIST', 'Food101', 'StandfordCars', 'C100', 'C10C', 'C100C', 'CUB2002011', 'INat2021Mini', 'CelebA'] # 'CatsDogs'

"""
class resized_dataset(Dataset):
    def __init__(self, dataset, transform=None, start=None, end=None, resize=None):
        self.data=[]
        if start == None:
            start = 0
        if end == None:
            end = dataset.__len__()
        if resize is None:
            for i in range(start, end):
                self.data.append((*dataset.__getitem__(i)))
        else:
            for i in range(start, end):
                item = dataset.__getitem__(i)
                self.data.append((F.center_crop(F.resize(item[0],resize,Image.BILINEAR), resize), item[1]))
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.transform:
            return (self.transform(self.data[idx][0]), self.data[idx][1], idx)
        else:
            return self.data[idx], idx

"""

class C10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(C10, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class C100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(C100, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]

class C10C(datasets.VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download = False):
        super(C10C, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        corruptions = load_txt(f'{root}/corruptions.txt')

        self.data = []
        self.targets = []

        target_file = f"{root}/labels.npy"
        for c in corruptions:
            #print("Extraction for test - corruption ", c)
            img_file = f"{root}/{c}.npy"
            data = np.load(img_file)
            targets = np.load(target_file)
            self.data.extend(data)
            self.targets.extend(targets)

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

class C100C(datasets.VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download = False):
        super(C100C, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        corruptions = load_txt(f'{root}/corruptions.txt')

        self.data = []
        self.targets = []

        target_file = f"{root}/labels.npy"
        for c in corruptions:
            #print("Extraction for test - corruption ", c)
            img_file = f"{root}/{c}.npy"
            data = np.load(img_file)
            targets = np.load(target_file)
            self.data.extend(data)
            self.targets.extend(targets)

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

class SVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(SVHN, self).__init__(root, split=split, transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class INat2021Mini(datasets.INaturalist):
    def __init__(self, root, version, target_type=None, transform=None, target_transform=None, download=False):
        super(INat2021Mini, self).__init__(root, version=version,
                                           transform=transform, target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = super(INat2021Mini, self).__getitem__(index)
        return img, target, index


class CUB2002011(datasets.VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CUB2002011, self).__init__(root, transform=transform,
                                         target_transform=target_transform)
        self.data = []
        self.targets = []

        path_train_test_split = f"{root}/train_test_split.txt"
        path_images_ids = f"{root}/images.txt"
        path_labels_ids = f"{root}/image_class_labels.txt"

        images_ids = {}
        with open(path_images_ids) as f_ids:
            for line in f_ids.readlines():
                line = line.replace("\n", "")
                images_ids[line.split(" ")[0]] = line.split(" ")[1]

        targets_dict = {}
        with open(path_labels_ids) as f_tids:
            for line in f_tids.readlines():
                line = line.replace("\n", "")
                targets_dict[line.split(" ")[0]] = line.split(" ")[1]

        def add_data_add_target(root, line, images_ids, targets_dict):
            data = np.asarray(Image.open(f"{root}/images/{images_ids[line[0]]}").convert("RGB"))
            tar = int(targets_dict[line[0]]) - 1 # targets from 0 to 199
            targets = np.asarray(tar)
            self.data.append(data)
            self.targets.append(targets)

        with open(path_train_test_split) as f:
            for line in f.readlines():
                line = line.replace("\n", "").split(" ")
                if train and line[1] == "1":
                    add_data_add_target(root, line, images_ids, targets_dict)
                elif not train and line[1] == "0":
                    add_data_add_target(root, line, images_ids, targets_dict)

        self.data = np.array(self.data, dtype=object)
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

class ImageNetKaggle(Dataset):
    def __init__(self, root, split=None, transform=None ):
        self.samples = []
        self.targets = []
        self.classes = [] # Strings names of the classes
        self.transform = transform
        self.syn_to_class = {}
        self.syn_to_strclass = {}

        nClasses = 1000 # Imagenet Subset

        with open(os.path.join(root, "imagenet_class_index.json").replace("\\", "/"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
                self.syn_to_strclass[v[1]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json").replace("\\", "/"), "rb") as f:
            self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split).replace("\\", "/")
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id).replace("\\", "/")
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample).replace("\\", "/")
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry).replace("\\", "/")
                self.samples.append(sample_path)
                self.targets.append(target)
        #print("self.syn_to_strclass", self.syn_to_strclass)
        self.classes = list(self.syn_to_strclass.keys())
        """
        all_classes_names = sorted(os.listdir(root))
        for i, name in enumerate(all_classes_names):
            folder_name = name
            folder_path = os.path.join(root, folder_name)
            file_names = os.listdir(folder_path)

            # Split:
            if split is not None:
                num_train = int(len(file_names) * 0.8) # 80% Training data
            for j, fid in enumerate(file_names):
                if split == 'train' and j >= num_train:  # ensures only the first 80% of data is used for training
                    break
                elif split == 'test' and j < num_train:  # skips the first 80% of data used for training
                    continue
                self.samples.append(os.path.join(folder_path, fid))
                self.labels.append(i)

        print(f"Dataset Size: {len(self.labels)}")
        self.targets = self.labels  # Sampler needs to use targets
        """

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return np.asarray(sample), label, index
        """
        x = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            x = self.transform(x)


        return x, self.targets[index], index


class CelebA(datasets.CelebA):
    def __init__(self, root, split='train', target_type='attr', transform=None, download=False):
        super(CelebA, self).__init__(root, split=split, transform=transform, target_type=target_type, download=download)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target[2], index


class MNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class Food101(datasets.Food101):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(Food101, self).__init__(root, split=split, transform=transform,
                                     target_transform=target_transform, download=download)


    def __getitem__(self, index):
        """
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index
        """
        img, target = super().__getitem__(index)
        return img, target, index


class StandfordCars(datasets.StanfordCars):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(StandfordCars, self).__init__(root, split=split, transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        """
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)


        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
        """
        img, target = super().__getitem__(index)
        return img, target, index