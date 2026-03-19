import torchvision.transforms as T
import random

def random_augment(image):
    augmentations = [
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomSolarize(threshold=128),
        T.RandomPosterize(bits=4),
        T.RandomAdjustSharpness(sharpness_factor=2),
        T.RandomAutocontrast(),
        T.RandomEqualize()
    ]
    funcs = random.sample(augmentations, 3)
    for func in funcs:
        image = func(image)
    return image
