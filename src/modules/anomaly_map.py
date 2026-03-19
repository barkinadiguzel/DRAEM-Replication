import torch
import torch.nn.functional as F

def image_level_score(Mo, local_pool_size=7):
    if Mo.dim() == 2:
        Mo = Mo.unsqueeze(0).unsqueeze(0)  
    elif Mo.dim() == 3:
        Mo = Mo.unsqueeze(0)              

    Mo_local = F.avg_pool2d(Mo, kernel_size=local_pool_size, stride=1, padding=local_pool_size//2)

    eta = Mo_local.max()

    return eta
