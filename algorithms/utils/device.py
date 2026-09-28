import os
import torch


def default_device():
    """Pick the compute device: $NEURALPLANE_DEVICE, else Apple Metal (MPS), else CUDA, else CPU."""
    override = os.environ.get("NEURALPLANE_DEVICE")
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def synchronize(device):
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def empty_cache(device):
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def allocated_memory_mb(device):
    device = torch.device(device)
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1024 ** 2
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device=device) / 1024 ** 2
    return 0.0
