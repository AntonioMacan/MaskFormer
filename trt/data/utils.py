import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from .dataset import CityscapesSimpleDataset, CITYSCAPES_COLORMAP


def prepare_data(root_path, subset='val', num_images=None, batch_size=1, image_size=(512, 1024)):    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.PILToTensor(),
    ])

    dataset = CityscapesSimpleDataset(root_path, subset, transform)

    if num_images and num_images < len(dataset):
        dataset = Subset(dataset, list(range(num_images)))
    
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        paths = [item['path'] for item in batch]
        labels = [item['label'] for item in batch]
        return images, paths, labels
    
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return loader


def save_segmentation_result(predictions, output_path):
    """
    Save segmentation predictions as a colored image
    
    Args:
        predictions: Numpy array of class predictions (H, W)
        output_path: Path to save the output image
    """
    h, w = predictions.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in enumerate(CITYSCAPES_COLORMAP):
        rgb[predictions == cls_id] = color

    # invalid trainIds -> black
    mask_valid = (predictions >= 0) & (predictions < len(CITYSCAPES_COLORMAP))
    rgb[~mask_valid] = (0, 0, 0)

    Image.fromarray(rgb).save(output_path)

def save_comparison_visualization(original_image, prediction, output_path):
    """
    Save a side-by-side visualization of original image and segmentation
    
    Args:
        original_image: PIL Image or numpy array
        prediction: Numpy array of class predictions (H, W)
        output_path: Path to save the output image
    """
    # Convert original image to numpy if it's a PIL image
    if not isinstance(original_image, np.ndarray):
        original_image = np.array(original_image)
    
    # Ensure original image is RGB
    if original_image.ndim==3 and original_image.shape[2]==3:
        # Convert CHW to HWC
        original_image = np.transpose(original_image, (1, 2, 0))
    
    # Convert segmentation to RGB
    h, w = prediction.shape
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls_id, color in enumerate(CITYSCAPES_COLORMAP):
        seg_rgb[prediction == cls_id] = color
    
    # Invalid trainIds ➜ black
    mask_valid = (prediction >= 0) & (prediction < len(CITYSCAPES_COLORMAP))
    seg_rgb[~mask_valid] = (0, 0, 0)
    
    # Resize original image if needed
    if original_image.shape[:2] != (h, w):
        from PIL import Image
        original_image = np.array(Image.fromarray(original_image).resize((w, h)))
    
    # Create side-by-side visualization
    combined = np.hstack([original_image, seg_rgb])
    Image.fromarray(combined).save(output_path)