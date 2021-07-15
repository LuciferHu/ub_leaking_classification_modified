import pandas as pd
import torch
from torch.utils.data import Dataset


class Ub_Leaking_Dataset(Dataset):
    def __init__(self, ub_leaking, transform=None):
        assert isinstance(ub_leaking, pd.DataFrame)
        assert len(ub_leaking.columns) == 3

        self.ub_leaking_df = ub_leaking
        self.transform = transform

    def __len__(self):
        return len(self.ub_leaking_df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        spectrogram, class_id, fold = self.ub_leaking_df.iloc[index]

        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        return {"spectrogram": spectrogram, "label": class_id}