from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torchvision
torchvision.disable_beta_transforms_warning()


from PIL import Image
import numpy as np



class ImageDataset(Dataset):
    """A PyTorch Dataset class to be used in a DataLoader to create batches.
    
    Attributes:
        dataset: A list of dictionaries containing 'label' and 'image' keys.
        classes: A list of class names.
        tfms: A torchvision.transforms.Compose object combining all the desired transformations.
    """
    def __init__(self, dataset, classes, tfms):
        self.dataset = dataset
        self.classes = classes
        self.tfms = tfms
        
    def __len__(self):
        """Returns the total number of samples in this dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Fetches a sample from the dataset at the given index.
        
        Args:
            idx: The index to fetch the sample from.
            
        Returns:
            A tuple of the transformed image and its corresponding label index.
        """
        sample = self.dataset[idx]
        image, label = sample['image'], sample['label']
        return self.tfms(image), label



class ImageDatasetWithoutHF(Dataset):
    """A PyTorch Dataset class to be used in a DataLoader to create batches.
    
    Attributes:
        dataset: A list of image paths.
        classes: A list of class names.
        tfms: A torchvision.transforms.Compose object combining all the desired transformations.
    """
    def __init__(self, img_paths, classes, tfms):
        self.img_paths = img_paths
        self.classes = classes
        self.tfms = tfms
        
        # Create a mapping from class names to class indices
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
    def __len__(self):
        """Returns the total number of samples in this dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Fetches a sample from the dataset at the given index.
        
        Args:
            idx: The index to fetch the sample from.
            
        Returns:
            A tuple of the transformed image and its corresponding label index.
        """
        # Get the path of the image at the given index
        img_path = self.img_paths[idx]
        
        # Get the label of the image
        label = self.class_to_idx[img_path.parent.name]
        
        # Open the image
        image = Image.open(img_path).convert('RGB')
        
        return self.tfms(image), label
    



from torchvision import transforms

from torch import Tensor
from typing import Dict, Tuple, List, Optional

# This class extends the TrivialAugmentWide class provided by PyTorch's transforms module.
# TrivialAugmentWide is an augmentation policy randomly applies a single augmentation to each image.
class CustomTrivialAugmentWide(transforms.TrivialAugmentWide):
    # The _augmentation_space method defines a custom augmentation space for the augmentation policy.
    # This method returns a dictionary where each key is the name of an augmentation operation and 
    # the corresponding value is a tuple of a tensor and a boolean value.
    # The tensor defines the magnitude of the operation, and the boolean defines  
    # whether to perform the operation in both the positive and negative directions (True)
    # or only in the positive direction (False).
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        
        # Define custom augmentation space
        custom_augmentation_space = {
            # Identity operation doesn't change the image
            "Identity": (torch.tensor(0.0), False),
            
            # Distort the image along the x or y axis, respectively.
            "ShearX": (torch.linspace(0.0, 0.25, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.25, num_bins), True),

            # Move the image along the x or y axis, respectively.
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),

            # Rotate operation: rotates the image.
            "Rotate": (torch.linspace(0.0, 45.0, num_bins), True),

            # Adjust brightness, color, contrast,and sharpness respectively.
            "Brightness": (torch.linspace(0.0, 0.75, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),

            # Reduce the number of bits used to express the color in each channel of the image.
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),

            # Invert all pixel values above a threshold.
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),

            # Maximize the image contrast by setting the darkest color to black and the lightest to white.
            "AutoContrast": (torch.tensor(0.0), False),

            # Equalize the image histogram to improve its contrast.
            "Equalize": (torch.tensor(0.0), False),
        }
        
        # The function returns the dictionary of operations.
        return custom_augmentation_space



import torchvision.transforms.functional as TF

class ResizePad(nn.Module):
    def __init__(self, max_sz=256, padding_mode='edge'):
        """
        A PyTorch module that resizes an image tensor and adds padding to make it a square tensor.

        Args:
        max_sz (int, optional): The size of the square tensor.
        padding_mode (str, optional): The padding mode used when adding padding to the tensor.
        """
        super().__init__()
        self.max_sz = max_sz
        self.padding_mode = padding_mode
        
    def forward(self, x):
        # Get the width and height of the image tensor
        w, h = TF.get_image_size(x)
        
        # Resize the image tensor so that its minimum dimension is equal to `max_sz`
        size = int(min(w, h) / (max(w, h) / self.max_sz))
        x = TF.resize(x, size=size, antialias=True)
        
        # Add padding to make the image tensor a square
        w, h = TF.get_image_size(x)
        offset = (self.max_sz - min(w, h)) // 2
        padding = [0, offset] if h < w else [offset, 0]
        x = TF.pad(x, padding=padding, padding_mode=self.padding_mode)
        x = TF.resize(x, size=[self.max_sz] * 2, antialias=True)
        
        return x
    

