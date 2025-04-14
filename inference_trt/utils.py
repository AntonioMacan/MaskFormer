import numpy as np
from PIL import Image
import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader


# Cityscapes color map
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


class CityscapesSimpleDataset:
    """Simplified Cityscapes dataset loader for inference"""
    
    def __init__(self, root_path, subset='val', transform=None):
        self.root_path = Path(root_path)
        self.subset = subset
        self.transform = transform
        self.images_dir = self.root_path / 'leftImg8bit' / subset
        self.image_paths = list(self.images_dir.glob('*/*_leftImg8bit.png'))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'path': str(img_path)
        }


def prepare_data(root_path, subset='val', num_images=None, batch_size=1, image_size=(512, 1024)):    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.0)
    ])
    
    dataset = CityscapesSimpleDataset(
        root_path, 
        subset=subset,
        transform=transform
    )
    
    if num_images and num_images < len(dataset):
        dataset = torch.utils.data.Subset(dataset, list(range(num_images)))
    
    # Simple collate function
    def collate_fn(batch):
        images = [{"image": item['image']} for item in batch]
        paths = [item['path'] for item in batch]

        return {
            'images': images,
            'paths': paths,
        }
    
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return loader


def save_segmentation_result(predictions, output_path):
    # Create a colored image from the segmentation map
    colored_pred = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)
    
    # Apply the colormap - only use the first 19 valid classes (trainId 0-18)
    # MaskFormer already outputs in trainId space
    for class_idx in range(min(19, len(CITYSCAPES_COLORMAP))):
        colored_pred[predictions == class_idx] = CITYSCAPES_COLORMAP[class_idx]
    
    # Handle any pixels with class labels outside the expected range
    mask_valid = (predictions < 19) & (predictions >= 0)
    colored_pred[~mask_valid] = (0, 0, 0)  # Set invalid pixels to black
    
    # Save as image
    colored_pred_img = Image.fromarray(colored_pred)
    colored_pred_img.save(output_path)