import torch

def get_torch_device(device_name=None):
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device name: "+device_name)
    return torch.device(device_name)

