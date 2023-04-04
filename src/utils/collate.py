import torch


def collate_fn(batch):
    if isinstance(batch[0], tuple):
        return tuple(collate_fn(list(x)) for x in zip(*batch))
    elif isinstance(batch[0], dict):
        return {key: collate_fn([x[key] for x in batch]) for key in batch[0]}
    elif isinstance(batch[0], torch.Tensor):
        return torch.stack(batch)
    elif isinstance(batch[0], list):
        if isinstance(batch[0][0], dict):
            return {
                key: collate_fn([collate_fn([x[key] for x in y]) for y in batch])
                for key in batch[0][0]
            }

    return torch.tensor(batch)
