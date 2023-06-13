import torch
import torch.nn.functional as F


def object_to_device(object, device):
    if isinstance(object, dict):
        for key in object:
            object[key] = object_to_device(object[key], device)
    else:
        object = object.to(device)

    return object


def add_first_dim(object):
    if isinstance(object, dict):
        for key in object:
            object[key] = object[key].unsqueeze(0)
    else:
        object = object.unsqueeze(0)

    return object


def get_user_repr_from_index(user_reprs, idx):
    if isinstance(user_reprs, tuple):
        return tuple(vector[idx].unsqueeze(0) for vector in user_reprs)
    else:
        return user_reprs[idx].unsqueeze(0)


def masked_softmax(tensor: torch.Tensor, mask: torch.Tensor, **kwargs):
    return F.softmax(tensor - 100 * (1 - mask), **kwargs)
