import os
import torch

file = os.path.expanduser("~/.cache/yolox-burn/yolox_tiny.pth")


# Load the state dictionary from the .pth file
checkpoint = torch.load(file, map_location=torch.device("cpu"))

# Print the structure of the loaded state dictionary
print("Model structure (layer names and parameter shapes):")
for name, param in checkpoint["model"].items():
    print(f"{name}")
