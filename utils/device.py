import torch


def get_device(cuda_enable):
    if cuda_enable:
        return torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device('cpu')
