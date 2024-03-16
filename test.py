# Import the resnet module
from timm.models import vision_transformer as model_family

# Define the base model variant to use
base_model = 'vit_small_patch32_224'
version = "augreg_in1k"

# Get the default configuration of the chosen model
model_cfg = model_family.default_cfgs[base_model].default.to_dict()

# Show the default configuration values
print(model_cfg)


# resnet50.a3_in1k 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)

# swin_s3_tiny_224.ms_in1k 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)

# vit_base_patch32_224.augreg_in1k 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)