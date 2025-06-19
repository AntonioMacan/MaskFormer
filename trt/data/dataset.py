from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np


# Map original Cityscapes ID to training ID [0-18], everything else -> 255
CITYSCAPES_ID_TO_TRAINID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
    17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 31: 16, 32: 17, 33: 18
}

CITYSCAPES_COLORMAP = [
    (128, 64, 128),   # road
    (244, 35, 232),   # sidewalk
    (70, 70, 70),     # building
    (102, 102, 156),  # wall
    (190, 153, 153),  # fence
    (153, 153, 153),  # pole
    (250, 170, 30),   # traffic light
    (220, 220, 0),    # traffic sign
    (107, 142, 35),   # vegetation
    (152, 251, 152),  # terrain
    (70, 130, 180),   # sky
    (220, 20, 60),    # person
    (255, 0, 0),      # rider
    (0, 0, 142),      # car
    (0, 0, 70),       # truck
    (0, 60, 100),     # bus
    (0, 80, 100),     # train
    (0, 0, 230),      # motorcycle
    (119, 11, 32),    # bicycle
]


class RemapLabels:
    def __init__(self, mapping: dict, ignore_id=255, total=256):
        self.mapping = np.ones((total,), dtype=np.uint8) * ignore_id
        for k, v in mapping.items():
            self.mapping[k] = v
        self.ignore_id = ignore_id

    def __call__(self, label: torch.Tensor) -> torch.Tensor:
        # label: [H, W] torch tensor, dtype=long
        np_label = label.numpy()
        remapped = self.mapping[np_label]
        return torch.from_numpy(remapped).long()


class CityscapesSimpleDataset(Dataset):
    def __init__(self, root_path, subset='val', transform=None):
        self.root_path = Path(root_path)
        self.subset = subset
        self.transform = transform

        self.images_dir = self.root_path / 'leftImg8bit' / subset
        self.labels_dir = self.root_path / 'gtFine' / subset

        self.image_paths = list(self.images_dir.glob('*/*_leftImg8bit.png'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        sample = {'image': image, 'path': str(img_path)}

        label_path = str(img_path).replace('leftImg8bit', 'gtFine', 1).replace('_leftImg8bit.png', '_gtFine_labelIds.png')
        label = Image.open(label_path)
        label = torch.from_numpy(np.array(label)).long()
        label = RemapLabels(CITYSCAPES_ID_TO_TRAINID)(label)

        sample['label'] = label

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample