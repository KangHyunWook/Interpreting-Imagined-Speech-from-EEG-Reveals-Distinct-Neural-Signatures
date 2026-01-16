from torch.utils.data import Dataset, DataLoader
from create_dataset import BCIComp,FEIS

import torch
import numpy as np

class EEGDataset(Dataset):
    def __init__(self, config):
        if 'track#3' in config.dataset_dir.lower():
            dataset=BCIComp(config)
        elif 'feis' in config.dataset_dir.lower():
            dataset=FEIS(config)
            
        self.data = dataset.get_data(config.mode)
        self.len = len(self.data)
    

    def __getitem__(self, index):
        return self.data[index]  
    
    def __len__(self):
        return self.len  
        

def get_loader(config, shuffle=True):
    dataset = EEGDataset(config)
    
    def collate_fn(batch):

        #todo:
        features = torch.cat([torch.FloatTensor(np.asarray([sample[0]])) for sample in batch])
        labels = torch.cat([torch.from_numpy(np.asarray([sample[1]])) for sample in batch])   
        
        return features, labels
        
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)
        
    return data_loader