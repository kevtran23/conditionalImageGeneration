import numpy as np
import torch
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import *


class CaptionedImageDataset(Dataset):
    def __init__(self, image_shape, n_classes):
        self.image_shape = image_shape
        self.n_classes = n_classes

    def __getitem__(self, index: int) -> (torch.tensor, torch.tensor, list):
        '''
        :param index: index of the element to be fetched
        :return: (image : torch.tensor , class_ids : torch.tensor ,captions : list(str))
        '''
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)




class Imagenet32Dataset(CaptionedImageDataset):
    def __init__(self, root="datasets/ImageNet32", train=True, max_size=-1):
        '''
        :param dirname: str, root dir where the dataset is downloaded
        :param train: bool, true if train set else val
        :param max_size: int, truncate size of the dataset, useful for debugging
        '''
        super().__init__((3, 32, 32), 1000)
        self.root = root

        if train:
            self.dirname = os.path.join(root, "train")
        else:
            self.dirname = os.path.join(root, "val")

        self.classId2className = load_vocab_imagenet(os.path.join(root, "map_clsloc.txt"))
        data_files = sorted(os.listdir(self.dirname))
        self.images = []
        self.labelIds = []

        for i, f in enumerate(data_files):
            print("loading data file {}/{}, {}".format(i + 1, len(data_files), os.path.join(self.dirname, f)))
            data = np.load(os.path.join(self.dirname, f))
            self.images.append(data['data'])
            self.labelIds.append(data['labels'] - 1)
        self.images = np.concatenate(self.images, axis=0)
        self.labelIds = np.concatenate(self.labelIds)
        self.labelNames = [self.classId2className[y-1] for y in self.labelIds]

        if max_size >= 0:
            # limit the size of the dataset
            self.labelNames = self.labelNames[:max_size]
            self.labelIds = self.labelIds[:max_size]

    def __getitem__(self, index: int) -> (torch.tensor, torch.tensor, list):
        image = torch.tensor(self.images[index]).reshape(3, 32, 32).float() / 128 - 1
        label = self.labelIds[index]
        caption = self.labelNames[index].replace("_", " ")
        return (image, label, caption)

    def __len__(self):
        return len(self.labelNames)


class CIFAR10Dataset(CaptionedImageDataset):
    def __init__(self, root='datasets/CIFAR10', train=True, max_size=-1):
        super().__init__((3, 32, 32), 10)
        self.dataset = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=ToTensor())
        self.max_size = max_size if max_size > 0 else len(self.dataset)
        self.text_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __len__(self):
        return self.max_size

    def __getitem__(self, item):
        img, label = self.dataset[item]
        img = 2 * img - 1
        text_label = self.text_labels[label]
        return img, label, text_label




def load_vocab_imagenet(vocab_file):
    vocab = {}
    with open(vocab_file) as f:
        for l in f.readlines():
            _, id, name = l[:-1].split(" ")
            vocab[int(id) - 1] = name.replace("_", " ")
    return vocab


if __name__ == "__main__":
    print("Testing CIFAR dataloader")
    d = CIFAR10Dataset()
    for i in range(2):
        i = np.random.randint(0, len(d))
        img, class_label, text = d[i]
        img = np.transpose(img.reshape((3, 32, 32)), [1, 2, 0])
        plt.figure(figsize=(1.5, 1.5))
        plt.imshow(img)
        plt.title(text)
        plt.show()

    print("Testing Imagenet32 dataloader")
    d = Imagenet32Dataset()
    for i in range(2):
        i = np.random.randint(0, len(d))
        img, class_label, text = d[i]
        img = np.transpose(img.reshape((3, 32, 32)), [1, 2, 0])
        plt.figure(figsize=(1.5, 1.5))
        plt.imshow(img)
        plt.title(text)
        plt.show()
