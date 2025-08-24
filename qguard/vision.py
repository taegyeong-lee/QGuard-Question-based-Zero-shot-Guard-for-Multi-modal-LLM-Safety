from typing import List, Optional
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

# ImageNet statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _find_closest_aspect_ratio(aspect_ratio: float, target_ratios, width: int, height: int, image_size: int):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def _dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if (i * j) <= max_num and (i * j) >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    tgt = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * tgt[0]
    target_height = image_size * tgt[1]
    blocks = tgt[0] * tgt[1]

    resized = image.resize((target_width, target_height))
    tiles = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))

    return tiles

def load_image_to_pixel_values(image_file: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(tile) for tile in tiles]) 
    return pixel_values

def load_images_batch(
    image_files: Optional[List[str]],
    input_size: int = 448,
    max_num: int = 12,
) -> Optional[torch.Tensor]:
    if not image_files:
        return None
    batches = []
    for p in image_files:
        try:
            pv = load_image_to_pixel_values(p, input_size=input_size, max_num=max_num)
            if pv is not None and pv.numel() > 0:
                batches.append(pv)
        except Exception:
            continue
    if not batches:
        return None
    return torch.cat(batches, dim=0) 
