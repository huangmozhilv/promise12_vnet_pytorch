import pdb
import torch
import torch.utils.data as data

import numpy as np

class customDataset(data.Dataset):
    '''
    For medical segmentation decathlon.
    '''

    def __init__(self, mode, images, GT, transform=None, GT_transform=None):
        if images is None:
            raise(RuntimeError("images must be set"))
        self.mode = mode
        self.images = images
        self.GT = GT
        self.transform = transform
        self.GT_transform = GT_transform

    def __getitem__(self, index):
        """
        Args:
            index(int): Index
        Returns:
            tuple: (image, GT) where GT is index of the
        """
        if self.mode == "train":
            # keys = list(self.images.keys())
            # id = keys[index]
            # because of data augmentation, train images are stored in a 4-d array, with first d as sample index.
            image = self.images[index]
            # print("image shape from DataManager shown in PROMISE12:" + str(image.shape)) # e.g. 96,96,48.
            image = np.transpose(image,[2,1,0]) # added by Chao
            image = np.expand_dims(image, axis=0)
            # print("expanded image dims:{}".format(str(image.shape)))
            # pdb.set_trace()
            image = image.astype(np.float32)
            if self.transform is not None:
                image = torch.from_numpy(image)
                # image = self.transform(image)

            GT = self.GT[index]
            GT = np.transpose(GT, [2, 1, 0])
            if self.GT_transform is not None:
                GT = self.GT_transform(GT)
            return image, GT
        elif self.mode == "test":
            keys = list(self.images.keys())
            id = keys[index]
            image = self.images[id]
            image = np.transpose(image, [2, 1, 0])  # added by Chao
            image = np.expand_dims(image, axis=0)
            # print("expanded image dims:{}".format(str(image.shape)))
            # pdb.set_trace()
            image = image.astype(np.float32)
            if self.transform is not None:
                image = torch.from_numpy(image)
                # image = self.transform(image)

            GT = self.GT[id+'_segmentation'] # require customization
            GT = np.transpose(GT, [2, 1, 0])
            if self.GT_transform is not None:
                GT = self.GT_transform(GT)
            return image, GT, id
        elif self.mode == "infer":# added by Chao
            keys = list(self.images.keys())
            id = keys[index]
            image = self.images[id]
            # print("image shape from DataManager shown in PROMISE12:" + str(image.shape))
            image = np.transpose(image,[2,1,0]) # added by Chao
            image = np.expand_dims(image, axis=0)
            image = image.astype(np.float32)
            return image, id

    def __len__(self):
        return len(self.images)
