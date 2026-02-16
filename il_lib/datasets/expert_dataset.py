import pandas as pd
import torch


class ExpertDataset():
    def __init__(self, file_path, obs_columns, act_columns, device=None, normalise= False):
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        obs_np = df[obs_columns].values.astype("float32")
        act_np = df[act_columns].values.astype("float32")

        self.obs = torch.tensor(obs_np)
        self.acts = torch.tensor(act_np)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self,idx):
        return self.obs[idx], self.acts[idx]
    



    

