class Config:
    lambda_ssim = 0.85      
    beta_min = 0.1 # the anomaly's participation rate 
    beta_max = 1.0 # the anomaly's participation rate 
    gamma_focal = 2.0
    fsf_size = 21 # smoothing parameter
    img_channels = 3
    img_height = 256
    img_width = 256
