"""
@User: sandruskyi
Prepare dataset. Used in main.py
"""
import os
import numpy as np
import torch
from torch.utils.data import random_split, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import training_functions.dataset_utils as dataset_utils
from training_functions.custom_dataset import CustomDataset, CustomDataset_newCacophony


__all__ = [ 'prepare_dataset', 'reduce_channels', 'fpr_remove', 'convert_binary', 'get_train_val_samplers', 'setup_binary']



def reduce_channels(x, num_channels):
    """Remove a given list of channels from the video data"""
    if len(num_channels) == 1:
        return x[:, :, num_channels[0]:num_channels[0]+1, :, :]
    return x[:, :, num_channels[0]:num_channels[1], :, :]

def fpr_remove(x, y, ids, proportion=0.6):
    """Removes false positives from the dataset. Can help model performance"""
    fpr_label = 4
    fp_indices = np.nonzero(y == fpr_label)[0]
    generator = np.random.default_rng(1)
    to_select = int(len(fp_indices) * proportion)
    to_remove = generator.choice(fp_indices, to_select, replace=False)
    inverse = np.ones(y.shape[0])
    inverse[to_remove] = 0
    inverse = np.nonzero(inverse)[0]
    return x[inverse], y[inverse], ids[inverse]

def convert_binary(y, one_values):
    """Convert a given dataset to binary"""
    output = np.zeros(y.shape)
    one_indices = []
    for v in one_values:
        one_indices.extend(np.nonzero(y == v)[0].tolist())
    output.put(one_indices, 1)
    return output

def load_data(debug=False, fpr=False, base_path="", npz_path="./", yolo=False, normalization=False):
    """Loads training data"""

    if yolo:
        file = f"{npz_path}newest_fpr.npz" if fpr else f"C:/Users/{SET_USER}/datasets/sfti/newest_with_ids_yolo.npz" # CHANGE THIS
    elif normalization:
        file = f"{npz_path}newest_fpr.npz" if fpr else f"C:/Users/{SET_USER}/datasets/sfti/re-normalised-with-ids.npz" # CHANGE THIS
    else:
        file = f"{npz_path}newest_fpr.npz" if fpr else f"C:/Users/{SET_USER}/datasets/sfti/newest.with_ids.npz" # CHANGE THIS
    print(file)
    x = "x" if fpr or yolo or normalization else "X"
    labels = "labels" if fpr or yolo or normalization else "label_map"
    data_quantity = 50
    if debug:
        with np.load(file) as all_data:
            x_train = all_data[f'{x}_train'][:data_quantity]
            y_train = all_data['y_train'][:data_quantity]
            x_val = all_data[f'{x}_val'][:data_quantity]
            y_val = all_data['y_val'][:data_quantity]
            x_test = all_data[f'{x}_test'][:data_quantity]
            y_test = all_data['y_test'][:data_quantity]
            train_ids = all_data['ids_train'][:data_quantity]
            val_ids = all_data['ids_val'][:data_quantity]
            test_ids = all_data['ids_test'][:data_quantity]
            labels = all_data[labels]
    else:
        with np.load(file) as all_data:
            x_train = all_data[f'{x}_train']
            y_train = all_data['y_train']
            x_val = all_data[f'{x}_val']
            y_val = all_data['y_val']
            x_test = all_data[f'{x}_test']
            y_test = all_data['y_test']
            labels = all_data[labels]
            train_ids = all_data['ids_train']
            val_ids = all_data['ids_val']
            test_ids = all_data['ids_test']
    return x_train, y_train, x_val, y_val, x_test, y_test, train_ids, val_ids, test_ids, labels


def shuffle_data(x, y, seed=0):
    """ Function to mix data based on a predefinided seed"""
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    return x_shuffled, y_shuffled

def setup_binary(debug, channels_last=True, fpr=True, reduce_channel=True, npz_path="./", yolo=False, normalization=False):
    """Combine many functions for creating the binary case"""
    x_train, y_train, x_val, y_val, x_test, y_test, train_ids, val_ids, test_ids, labels = load_data(debug, fpr=False, npz_path=npz_path, yolo=yolo, normalization=normalization)
    if fpr:
        x_train, y_train, train_ids = fpr_remove(x_train, y_train, train_ids)
        x_val, y_val, val_ids = fpr_remove(x_val, y_val, val_ids)
    if x_train.shape[2] > 1 and reduce_channel:
        x_train = reduce_channels(x_train, [0, 3])
        x_val = reduce_channels(x_val, [0, 3])
        x_test = reduce_channels(x_test, [0, 3])
    predators = "mustelid, possum, rodent, cat, dog, hedgehog, leporidae, wallaby".split(", ")

    p_label_i = np.array([np.nonzero(labels == x) for x in predators]).flatten()
    y_train = convert_binary(y_train, p_label_i)
    y_val = convert_binary(y_val, p_label_i)
    y_test = convert_binary(y_test, p_label_i)
    if channels_last:
        x_train = np.moveaxis(x_train, 2, -1)
        x_val = np.moveaxis(x_val, 2, -1)
        x_test = np.moveaxis(x_test, 2, -1)

    labels = ["Non-predator", "Predator"]

    x_train, y_train = shuffle_data(x_train, y_train)
    x_val, y_val = shuffle_data(x_val, y_val)

    return x_train, y_train, x_val, y_val, x_test, y_test, train_ids, val_ids, test_ids, labels




def get_train_val_samplers(trainset, valid_size, seed, shuffle=True):
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return train_sampler, valid_sampler

# Custom Wrapper to include indices in the smaller datasets for the hyperparameter tunning
class ResetIndexDatasetHyperparameterTunning(Dataset):
    def __init__(self, dataset_subset):
        # dataset_subset is a Subset object with two variables: dataset and indices
        self.dataset_subset = dataset_subset # Has indices and dataset
        self.dataset = dataset_subset.dataset # Ref to the original dataset. It is the complete dataset, and with dataset_subset.indices we can access to the subset of this complete dataset.

    def __len__(self):
        return len(self.dataset_subset.indices)

    def __getitem__(self, idx):
        # Map the new index with the original Subset index
        original_idx = self.dataset_subset.indices[idx]
        return self.dataset[original_idx][0], self.dataset[original_idx][1], idx

def split_big_dataset_hyperparameter_tunning(dataset:datasets, split_size:float=0.8, seed:int=42):
    # Create split lengths based on original dataset lengths:
    split_length = int(len(dataset)*split_size)
    if split_size==0.8:
        remaining_length = len(dataset) - split_length
        train_split, val_split = random_split(dataset, lengths=[split_length, remaining_length], generator= torch.manual_seed(seed))
    elif split_size==0.2:
        remaining_length = len(dataset) - split_length - split_length
        train_split, val_split, test_split = random_split(dataset, lengths=[split_length, split_length, remaining_length], generator= torch.manual_seed(seed))
    else:
        exit("SPLIT SIZE SHOULD BE 0.8 OR 0.2")

    # We only want train and val
    train_split = ResetIndexDatasetHyperparameterTunning(train_split)
    val_split = ResetIndexDatasetHyperparameterTunning(val_split)

    return train_split, val_split

def prepare_dataset(dataset="cacophony", num_classes=2, args=None, seed=42, valid_size=0.1):
    """Download/charge the dataset and prepare it to use it"""
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    if dataset == "celebAadam":
        dataset = "celebA"

    dataset_name = dataset
    if dataset == 'cacophony2':
        # For cacophony dataset also

        num_classes = 2
        labels = ["Non-predator", "Predator"]

        MEMMAP_PATH = f"/data/{SET_USER}/datasets/memmap/" # CHANGE THIS

        if not os.path.exists(MEMMAP_PATH):
            raise ValueError("MEMMAP_PATH doesn't exist")

        x_train_shape = [238994, 9, 120, 160]
        x_val_shape = [26302, 9, 120, 160]
        x_test_shape = [25659, 9, 120, 160]

        # add 3 channels into the shape
        x_train_shape.insert(2, 3)
        x_val_shape.insert(2, 3)
        x_test_shape.insert(2, 3)

        y_train_shape = x_train_shape[0:1]
        y_val_shape = x_val_shape[0:1]
        y_test_shape = x_test_shape[0:1]

        x_train = torch.from_numpy(np.memmap(f'{MEMMAP_PATH}x_train.npy', dtype=np.uint8, mode="r", shape=tuple(x_train_shape)))
        y_train = torch.from_numpy(np.memmap(f'{MEMMAP_PATH}y_train.npy', dtype=np.uint8, mode="r", shape=tuple(y_train_shape)))

        x_val = torch.from_numpy(np.memmap(f'{MEMMAP_PATH}x_val.npy', dtype=np.uint8, mode="r", shape=tuple(x_val_shape)))
        y_val = torch.from_numpy(np.memmap(f'{MEMMAP_PATH}y_val.npy', dtype=np.uint8, mode="r", shape=tuple(y_val_shape)))

        x_test = torch.from_numpy(np.memmap(f'{MEMMAP_PATH}x_test.npy', dtype=np.uint8, mode="r", shape=tuple(x_test_shape)) / 255.0)
        y_test = torch.from_numpy(np.memmap(f'{MEMMAP_PATH}y_test.npy', dtype=np.uint8, mode="r", shape=tuple(y_test_shape)) / 1.0)

        # Create tensors
        trainset = CustomDataset_newCacophony(torch.Tensor(x_train), torch.Tensor(y_train), tuple(x_train_shape), tuple(y_train_shape))
        valset = CustomDataset_newCacophony(torch.Tensor(x_val), torch.Tensor(y_val), tuple(x_val_shape), tuple(y_val_shape))
        testset = CustomDataset_newCacophony(torch.Tensor(x_test), torch.Tensor(y_test), tuple(x_test_shape), tuple(y_test_shape))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=8, persistent_workers=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.train_batch, shuffle=True, num_workers=8, persistent_workers=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.train_batch, shuffle=True, num_workers=8, persistent_workers=True)

        train_batch = args.train_batch
        val_batch = args.train_batch
        test_batch = args.train_batch

        input_size = x_train_shape[1:]
        input_shape = input_size
        return trainloader, valloader, testloader, trainset, valset, testset, num_classes, train_batch, val_batch, test_batch, labels, input_size, input_shape

    elif dataset == 'cacophony':
        num_classes = 2
        input_size = (45, 3, 24, 24)
        input_shape = input_size
        x_train, y_train, x_val, y_val, x_test, y_test, train_ids, val_ids, test_ids, labels = setup_binary(False,
                                                                                                            channels_last=False,
                                                                                                            reduce_channel=True,
                                                                                                            fpr=True,
                                                                                                            normalization=True)
        trainset = CustomDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        valset = CustomDataset(torch.Tensor(x_val), torch.Tensor(y_val))
        testset = CustomDataset(torch.Tensor(x_test), torch.Tensor(y_test))


        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=False,
                                                  num_workers=args.workers) # It was shuffle in setup_binary taking into account the predefined np seed
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch, shuffle=False,
                                                num_workers=args.workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False,
                                                 num_workers=args.workers)

        train_batch = args.train_batch
        val_batch = args.test_batch
        test_batch = args.test_batch

        return trainloader, valloader, testloader, trainset, valset, testset, num_classes, train_batch, val_batch, test_batch, labels, input_size, input_shape
    elif dataset == 'cifar10' or dataset == 'cifar10C':
        # With ARCH=vgg16_bn
        # dataset = datasets.CIFAR10
        dataset = dataset_utils.C10
        num_classes = 10
        if args.transfer_learning:
            input_size = 224
        else:
            input_size = 32 # 32x32x3
        input_shape = (3, input_size, input_size)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        if args.transfer_learning:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop((input_size, input_size), scale=(0.05, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            if args.not_augmentation:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        if args.cluster:
            path_cifar = f'/data/{SET_USER}/datasets/CIFAR10' # CHANGE THIS
        else:
            path_cifar = '~/datasets/CIFAR10'
        trainset = dataset(root=path_cifar, train=True, download=True, transform=transform_train)
        validset = dataset(root=path_cifar, train=True, download=True, transform=transform_test)
        if dataset_name == 'cifar10':
            testset = dataset(root=path_cifar, train=False, download=True, transform=transform_test)
        else:
            if args.cluster:
                path_cifar = f'/data/{SET_USER}/datasets/CIFAR10C' # CHANGE THIS
            else:
                path_cifar = '~/datasets/CIFAR10C'
            dataset = dataset_utils.C10C
            testset = dataset(root=path_cifar, train=False, download=True, transform=transform_test)

        # Extracting class labels
        labels = trainset.classes



    elif dataset == 'cifar100' or dataset == 'cifar100C':
        # With ARCH=vgg16_bn
        # dataset = datasets.CIFAR100
        dataset = dataset_utils.C100
        num_classes = 100
        if args.transfer_learning:
            input_size = 224
            print("DOING TRANSFER LEARNING, input_size=", input_size)
        else:
            input_size = 32 # 32x32x3
        input_shape = (3, input_size, input_size)
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        if args.transfer_learning:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop((input_size, input_size), scale=(0.05, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        if args.cluster:
            path_cifar = f'/data/{SET_USER}/datasets/CIFAR100' # CHANGE THIS
        else:
            path_cifar = '~/datasets/CIFAR100'
        trainset = dataset(root=path_cifar, train=True, download=True, transform=transform_train)
        validset = dataset(root=path_cifar, train=True, download=True, transform=transform_test)
        if dataset_name == 'cifar100':
            testset = dataset(root=path_cifar, train=False, download=True, transform=transform_test)
        else:
            if args.cluster:
                path_cifar = f'/data/{SET_USER}/datasets/CIFAR100C' # CHANGE THIS
            else:
                path_cifar = '~/datasets/CIFAR100C'
            dataset = dataset_utils.C100C
            testset = dataset(root=path_cifar, train=False, download=True, transform=transform_test)

        # Extracting class labels
        labels = trainset.classes
    elif dataset == 'inat':
        dataset = dataset_utils.INat2021Mini
        num_classes = 10000
        input_size = 224 # Resized from
        input_shape = (3, input_size, input_size)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if args.cluster:
            path_inat = f'/data/{SET_USER}/datasets/INaturalist' # CHANGE THIS
        else:
            path_inat = '~/datasets/INaturalist'
        trainset = dataset(root=(path_inat+'/2021_train_mini'), version="2021_train_mini", download=False, transform=transform_train)
        validset = dataset(root=(path_inat+'/2021_train_mini'), version="2021_train_mini",  download=False, transform=transform_test)
        testset = dataset(root=(path_inat+'/2021_valid'), version="2021_valid", download=False, transform=transform_test)

        # Extracting class labels
        labels = [i for i in range(num_classes)]
    elif dataset == 'svhn':
        # With ARCH=vgg16_bn
        # dataset = datasets.SVHN
        dataset = dataset_utils.SVHN
        num_classes = 10
        input_size = 32 # 32x32x3
        input_shape = (3, input_size, input_size)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform_train = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,  std)
        ])

        if args.cluster:
            path_svhn = f'/data/{SET_USER}/datasets/SVHN' # CHANGE THIS
        else:
            path_svhn = '~/datasets/SVHN'

        trainset = dataset(root=path_svhn, split='train', download=True, transform=transform_train)
        validset = dataset(root=path_svhn, split='train', download=True, transform=transform_test)
        testset = dataset(root=path_svhn, split='test', download=True, transform=transform_test)

        # Extracting class labels
        labels = [i for i in range(10)]
    elif dataset == 'celebA':
        dataset = dataset_utils.CelebA
        num_classes = 2
        input_size = 224  # Resized from
        input_shape = (3, input_size, input_size)

        mean = (0.5063486, 0.4258108, 0.38318512)
        std = (0.26577517, 0.24520662, 0.24129295)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if args.cluster:
            path_cel = f'/data/{SET_USER}/datasets/celeba' # CHANGE THIS
        else:
            path_cel = '~/datasets/celeba'
        trainset = dataset(root=(path_cel), split='train', target_type='attr', download=True, transform=transform_train)
        validset = dataset(root=(path_cel), split='valid', target_type='attr', download=True, transform=transform_test)
        testset = dataset(root=(path_cel), split='test', target_type='attr', download=True, transform=transform_test)

        print(f"Datasets lengths-> Train:{len(trainset)}, Valid:{len(validset)}, Test:{len(testset)}")

        # Extracting class labels
        labels = [0, 1]  # Labels from 0 to 199, in the .txt the labels go from 1 to 200.

    elif dataset == 'birds':
        dataset = dataset_utils.CUB2002011
        num_classes = 200
        input_size = 224  # Resized from
        input_shape = (3, input_size, input_size)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if args.cluster:
            path_cub = f'/data/{SET_USER}/datasets/cub-200-2011/CUB_200_2011' # CHANGE THIS
        else:
            path_cub = '~/datasets/cub-200-2011/CUB_200_2011'
        trainset = dataset(root=(path_cub), train=True, download=False, transform=transform_train)
        validset = dataset(root=(path_cub), train=True, download=False, transform=transform_test)
        testset = dataset(root=(path_cub), train=False, download=False, transform=transform_test)

        # Extracting class labels
        labels = [i for i in range(0, num_classes)] # Labels from 0 to 199, in the .txt the labels go from 1 to 200.
        """
    elif dataset == 'catsdogs':
        dataset = dataset_utils.CatsDogs
        num_classes = 2
        input_size = 64
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=6),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # resizing the images to 64 and center crop them, so that they become 64x64 squares
        trainset = dataset(root='~/datasets/cats_dogs', split='train', transform=transform_train,
                                          resize=64)
        testset = dataset(root='~/datasets/cats_dogs', split='test', transform=transform_test, resize=64)
        """

    elif dataset == 'imagenet':
        # https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
        # https://github.com/BorealisAI/towards-better-sel-cls/blob/main/large_dataset_utils.py
        # With ARCH=resnet34
        dataset = dataset_utils.ImageNetKaggle
        num_classes = 1000
        input_size = 224 # 224x224x3
        input_shape = (3, input_size, input_size)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if args.cluster:
            path_imagenet = f'/data/{SET_USER}/datasets/imagenet' # CHANGE THIS
        else:
            path_imagenet = f'C:/Users/{SET_USER}/datasets/imagenet' # CHANGE THIS
        trainset = dataset(root=path_imagenet, split='train', transform=transform_train)
        validset = dataset(root=path_imagenet, split='train', transform=transform_test)
        testset = dataset(root=path_imagenet, split='val', transform=transform_test)

        # Extracting class labels
        labels = trainset.classes
    elif dataset == 'mnist':
        # https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
        # With ARCH=resnet34
        dataset = dataset_utils.MNIST
        num_classes = 10
        input_size = 28 # 28 x 28 x 1
        input_shape = (1, input_size, input_size)

        mean = (0.5, )
        std =  (0.5, )
        transform_train_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = dataset(root='~/datasets/mnist', train=True, download=True, transform=transform_train_test)
        validset = dataset(root='~/datasets/mnist', train=True, download=True, transform=transform_train_test)
        testset = dataset(root='~/datasets/mnist', train=False, download=True, transform=transform_train_test)

        # Extracting class labels
        labels = trainset.classes

    elif dataset == 'food101':
        # With ARCH=resnet34
        dataset = dataset_utils.Food101
        num_classes = 101
        input_size = 224 # 512x512x3 but resized to 224
        input_shape = (3, input_size, input_size)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]) # transforms.RandomResizedCrop(32),
        transform_test = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]) # transforms.CenterCrop(32),
        if args.cluster:
            path_food101 = f'/data/{SET_USER}/datasets/food101' # CHANGE THIS
        else:
            path_food101 = f'C:/Users/{SET_USER}/datasets/food101' # CHANGE THIS
        trainset_full = dataset(root=path_food101, split='train', download=True, transform=transform_train)
        labels = trainset_full.classes # Extracting class labels
        if args.hypersearchSplit:
            print("""DATASET SPLIT""")
            trainset, validset = split_big_dataset_hyperparameter_tunning(trainset_full, 0.8, seed)
        else:
            trainset = trainset_full
            validset = dataset(root=path_food101, split='train', download=True, transform=transform_test)
        testset = dataset(root=path_food101, split='test', download=True, transform=transform_test)

    elif dataset == 'standfordCars':
        # With ARCH=resnet34
        dataset = dataset_utils.StandfordCars
        num_classes = 196
        input_size = 224 # 360×240x3 but resized to 224
        input_shape = (3, input_size, input_size)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(35),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomPosterize(bits=2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = dataset(root='~/datasets/standfordCars', split='train', download=True, transform=transform_train)
        validset = dataset(root='~/datasets/standfordCars', split='train', download=True, transform=transform_test)
        testset = dataset(root='~/datasets/standfordCars', split='test', download=True, transform=transform_test)

        # Extracting class labels
        labels = trainset.classes

    print("Creating loaders")

    if dataset_name == 'celebA':
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=False,
                                                 num_workers=args.workers)
        print(f"Trainloader created CelebA")
        valloader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch, shuffle=False,
                                                  num_workers=args.workers)
        print(f"Valloader created CelebA")
    else:
        train_sampler, valid_sampler = get_train_val_samplers(trainset, valid_size, seed, shuffle=True) # Same seed for all the shuffles
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, sampler=train_sampler,num_workers=args.workers)
        valloader = torch.utils.data.DataLoader(validset, batch_size=args.train_batch, sampler=valid_sampler,
                                                num_workers=args.workers)
        print("Trainloader created")
        print("Valloader created")

    ## As they only divide CIFAR10 in train and test
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False,
                                             num_workers=args.workers)
    print(f"Testloader created")

    train_batch = args.train_batch
    val_batch = args.train_batch
    test_batch = args.test_batch

    #if args.debug:
    print("###############")
    print("###############")
    print(dataset_name + " DATASET:")
    print("trainset", trainset)
    print("validset", validset)
    print("testset", testset)
    print("###############")
    print("###############")

    # Cacophony dataset has a different return
    return trainloader, valloader, testloader, trainset, validset, testset, num_classes, train_batch, val_batch, test_batch, labels, input_size, input_shape

if __name__ == "__main__":
    pass