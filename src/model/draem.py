import torch
import torch.nn as nn
from src.layers.reconstructive_net import ReconstructiveNet
from src.layers.discriminative_net import DiscriminativeNet
from src.data.anomaly_generator import generate_anomalous_image
from src.model.utils import image_level_score 

class DRAEM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reconstructive = ReconstructiveNet()
        self.discriminative = DiscriminativeNet()
        self.config = config

    def forward(self, I, texture_images=None, simulate=False):
        if simulate and texture_images is not None:
            Ia, Ma = generate_anomalous_image(I, texture_images, beta_range=(self.config.beta_min, self.config.beta_max))
        else:
            Ia = I
            Ma = None

        Ir = self.reconstructive(Ia)
        Ic = torch.cat([Ia, Ir], dim=1)  
        eta_max = image_level_score(Mo)

        return Ir, Mo, eta_max, Ma
