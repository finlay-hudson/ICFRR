import os
import numpy as np
from torch.utils.data import Dataset
from .utils.augs import random_transform, sketch_aug
import torchvision.transforms as transforms
import os.path as osp
from PIL import Image


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    assert got_img
    return img


class BaseDataset(Dataset):
    def __init__(self, split='train',
                 root_dir='../dataset/Sketchy/',
                 version='sketch_tx_000000000000_ready', zero_version='zeroshot1', transform=None, aug=None,
                 shuffle=False, first_n_debug=-1):

        self.root_dir = root_dir
        self.version = version
        self.split = split

        self.img_dir = self.root_dir

        if self.split == 'train':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_train.txt')
        elif self.split == 'val':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_test.txt')
        elif self.split == 'zero':
            file_ls_file = os.path.join(self.root_dir, zero_version, self.version + '_filelist_zero.txt')
        else:
            print('unknown split for dataset initialization: ' + self.split)
            return

        with open(file_ls_file, 'r') as fh:
            file_content = fh.readlines()

        self.file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
        self.labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
        if shuffle:
            self.shuffle()

        if first_n_debug > 0:
            idxs = []
            for label in np.unique(self.labels):
                idxs.extend(np.where(self.labels == label)[0][:first_n_debug])
            self.file_ls = self.file_ls[idxs]
            self.labels = self.labels[idxs]

        self.transform = transform
        self.aug = aug
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.dataset = [os.path.join(self.img_dir, f) for f in self.file_ls]

    def __len__(self):
        return len(self.labels)

    def get_path_and_label(self, idx):
        return os.path.join(self.img_dir, self.file_ls[idx]), self.labels[idx]

    def __getitem__(self, idx):
        img = read_image(self.dataset[idx])

        if self.transform is not None:
            img = self.transform(img)

        if self.aug == 'sketch':
            img = sketch_aug(img)
        elif self.aug == 'img':
            img = random_transform(img)
        else:
            pass

        img = self.normalize(img)
        label = self.labels[idx]

        return img, label, idx

    def shuffle(self):
        s_idx = np.random.shuffle(np.arange(len(self.labels)))
        self.file_ls = self.file_ls[s_idx]
        self.labels = self.labels[s_idx]


class SketchyDataset(BaseDataset):
    def __init__(self, split='train', root_dir='../datasets/sketch_based_image_retrieval/Sketchy/',
                 version='sketch_tx_000000000000_ready', zero_version='zeroshot1', transform=None, aug=None,
                 shuffle=False, first_n_debug=-1):
        super().__init__(split, root_dir, version, zero_version, transform, aug, shuffle, first_n_debug)


class TUBerlinDataset(BaseDataset):
    def __init__(self, split='train', root_dir="../datasets/sketch_based_image_retrieval/TUBerlin/",
                 version='png_ready', zero_version='zeroshot', transform=None, aug=False, shuffle=False,
                 first_n_debug=-1):
        super().__init__(split, root_dir, version, zero_version, transform, aug, shuffle, first_n_debug)
