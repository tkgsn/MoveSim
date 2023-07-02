from torch.utils.data import Dataset
import torch
import random
import numpy as np

def make_padded_collate(ignore_idx, start_idx, end_idx):

    def padded_collate(batch):
        # compute max_len
        max_len = max([len(x) for x in batch])
        inputs = []
        targets = []
        times = []
        target_times = []
        labels = []

        for trajectory in batch:

            input = [start_idx] + trajectory + [ignore_idx] * (max_len - len(trajectory))
            target = trajectory + [end_idx] + [ignore_idx] * (max_len - len(trajectory))

            inputs.append(input)
            targets.append(target)

        return {"input":torch.Tensor(inputs).long(), "target":torch.Tensor(targets).long()}

    return padded_collate


def make_padded_collate_for_GANs(end_idx):

    def padded_collate(batch):
        # compute max_len
        max_len = max([len(input) for input, _ in batch])
        inputs = []
        targets = []

        for input, target in batch:

            input = input + [end_idx] * (max_len - len(input))

            inputs.append(input)
            targets.append(target)

        return inputs, targets

    return padded_collate




class TrajectoryDataset(Dataset):
    #Init dataset
    def __init__(self, data, n_bins):
        # compute max seq len in one line
        self.seq_len = max([len(trajectory) for trajectory in data])

        self.n_locations = (n_bins+2)**2
        vocab = list(range(self.n_locations)) + ['<end>', '<ignore>', '<start>', '<oov>', '<mask>', '<cls>']
        self.vocab = {e:i for i, e in enumerate(vocab)} 
        
        #special tags
        self.START_IDX = self.vocab['<start>']
        self.IGNORE_IDX = self.vocab['<ignore>'] #replacement tag for tokens to ignore
        self.OUT_OF_VOCAB_IDX = self.vocab['<oov>'] #replacement tag for unknown words
        self.MASK_IDX = self.vocab['<mask>'] #replacement tag for the masked word prediction task
        self.CLS_IDX = self.vocab['<cls>']
        self.END_IDX = self.vocab['<end>']

        self.data = data
    
    def __str__(self):
        return self.dataset_name
        
    #fetch data
    def __getitem__(self, index):
        trajectory = self.data[index]
        return trajectory

    def __len__(self):
        return len(self.data)

    

class RealFakeDataset(Dataset):
    
    def __init__(self, real_data, fake_data):
        self.real_data = real_data
        self.num_real_data = len(real_data)
        self.fake_data = fake_data
        self.num_fake_data = len(fake_data)
        
    def __getitem__(self, index):
        
        if index >= self.num_real_data:
            target = 0
            input = self.fake_data[index-self.num_real_data]
        else:
            target = 1
            input = self.real_data[index]

        return input, target
        
    def __len__(self):
        return self.num_real_data + self.num_fake_data