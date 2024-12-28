import torch

def split_dataset(dataset, train_size, val_size, seed=42):
    train, val = torch.utils.data.random_split(
        dataset=dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    return train, val
