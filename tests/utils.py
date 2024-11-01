import torch

def is_similar(result, expected, rtol=0.01, atol=0.01):
    return result.shape == expected.shape and torch.allclose(result, expected, rtol, atol)
