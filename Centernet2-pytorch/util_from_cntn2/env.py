import torch

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
"""
PyTorch version as a tuple of 2 ints. Useful for comparison.
"""



