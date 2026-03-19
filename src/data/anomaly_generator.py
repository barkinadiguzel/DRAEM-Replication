import torch
import random
import numpy as np
from PIL import Image

def generate_perlin_noise(shape, scale=10):
    noise = np.random.rand(*shape).astype(np.float32)
    return torch.from_numpy(noise)

def generate_anomalous_image(I, texture_images, beta_range=(0.1, 1.0)):
    C,H,W = I.shape
    P = generate_perlin_noise((H,W))
    threshold = random.uniform(0.3,0.7)
    Ma = (P > threshold).float()
    A = random.choice(texture_images)
  
    if A.shape != I.shape:
        A = F.interpolate(A.unsqueeze(0), size=(H,W), mode='bilinear', align_corners=False).squeeze(0)

    beta = random.uniform(*beta_range)
    Ma_inv = 1 - Ma
    Ia = Ma_inv * I + beta * (Ma * A)

    return Ia, Ma
