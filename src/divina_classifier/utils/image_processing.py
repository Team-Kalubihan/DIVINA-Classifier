from torchvision import transforms
from PIL import Image
import torch

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image for model inference."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = get_transform()
    return transform(image).unsqueeze(0)  # Add batch dimension

def load_image(image_path: str) -> torch.Tensor:
    """Load an image from a path and preprocess it."""
    image = Image.open(image_path)
    return preprocess_image(image)
