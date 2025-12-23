import torch
from torch.utils.data import Dataset

class WikiDataset(Dataset):
    def __init__(self, ids_tensor: torch.Tensor):
        self.ids = ids_tensor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # For language modeling, input is shifted
        x = self.ids[idx]
        return x

def load_dataset(path: str) -> dict:
    return torch.load(path, weights_only=True)

def get_split(dataset: dict, split: str, modality: str) -> torch.Tensor:
    """
    modality: 'text' or 'sem'
    """
    key = "text_ids" if modality == "text" else "sem_ids"
    return dataset[split][key]
