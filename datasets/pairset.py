import os
import torch
import torch.nn as nn
from . import transforms
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision.transforms import ToPILImage
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg

__all__ = ['Pairset', 'Sumset']

class PairTransform():
    def __init__(self, dataset1, dataset2=None):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __call__(self, input1, target1, input2=None, target2=None, cpu=False, gpu=True):
        input1, target1 = self.dataset1.transforms(input1, target1, cpu=cpu, gpu=gpu)

        if self.dataset2 is None:
            return input1, target1, input1, target1

        input2, target2 = self.dataset2.transforms(input2, target2, cpu=cpu, gpu=gpu)

        return input1, target1, input2, target2

class SumTransform():
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __call__(self, image, label, cpu=False, gpu=True):
        image, label = self.dataset1.transforms(image, label, cpu=cpu, gpu=gpu)

        return image, label

class Pairset(VisionDataset):
    def __init__(
            self,
            dataset1,
            dataset2=None,
    ):
        super(Pairset, self).__init__(root=None)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.transforms = PairTransform(dataset1, dataset2)

    def __getitem__(self, index: int, cpu=True, gpu=False) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image_train, label_train, _ = self.dataset1.__getitem__(index %\
                                                self.dataset1.__len__(), cpu=cpu, gpu=gpu)
        if self.dataset2 is not None:
            image_valid, label_valid, _ = self.dataset2.__getitem__(index %\
                                                    self.dataset2.__len__(), cpu=cpu, gpu=gpu)
        else:
            image_valid, label_valid = image_train, label_train

        return image_train, label_train, image_valid, label_valid, index, index

    def __weights__(self) -> torch.Tensor:
        return self.dataset1.__weights__()

    def __len__(self) -> int:
        if self.dataset2 is not None:
            return max(self.dataset1.__len__(), self.dataset2.__len__())
        else:
            return self.dataset1.__len__()

    def __outshape__(self) -> list:
        return self.dataset1.__outshape__()

    def __numclass__(self) -> int:
        return self.dataset1.__numclass__()

    def __numinput__(self) -> int:
        return self.dataset1.__numinput__()

    def save_output(self, root, outputs, targets, indices):
        self.dataset1.save_output(root, outputs, targets, indices)

# class Catset(VisionDataset):
#     def __init__(
#             self,
#             dataset1,
#             dataset2,
#     ):
#         super(Catset, self).__init__(root=None)
#         self.dataset1 = dataset1
#         self.dataset2 = dataset2

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is the image segmentation.
#         """

#         image, label, _ = self.dataset1.__getitem__(index)

#         for i in range(0,20):
#             _, guide, _ = self.dataset2.__getitem__(i)
#             image = torch.cat([image, guide], 0)

#         return image, label, index

#     def __weights__(self) -> torch.Tensor:
#         return self.dataset1.__weights__()

#     def __len__(self) -> int:
#         return self.dataset1.__len__()

#     def __outshape__(self) -> list:
#         return self.dataset1.__outshape__()

#     def __numclass__(self) -> int:
#         return self.dataset1.__numclass__()

#     def __numinput__(self) -> int:
#         return self.dataset1.__numinput__() + 20

#     def save_output(self, root, outputs, targets, indices):
#         self.dataset1.save_output(root, outputs, targets, indices)

class Sumset(VisionDataset):
    def __init__(
            self,
            dataset1,
            dataset2,
    ):
        super(Sumset, self).__init__(root=None)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.transforms = SumTransform(dataset1, dataset2)

    def __getitem__(self, index: int, cpu=True, gpu=False) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        len1 = self.dataset1.__len__()
        if index < len1:
            image, label, _ = self.dataset1.__getitem__(index, cpu=cpu, gpu=gpu)
        else:
            image, label, _ = self.dataset2.__getitem__(index - len1, cpu=cpu, gpu=gpu)

        return image, label, index

    def __weights__(self) -> torch.Tensor:
        return self.dataset1.__weights__()

    def __len__(self) -> int:
        return self.dataset1.__len__() + self.dataset2.__len__()

    def __numclass__(self) -> int:
        return self.dataset1.__numclass__()

    def __numinput__(self) -> int:
        return self.dataset1.__numinput__()
