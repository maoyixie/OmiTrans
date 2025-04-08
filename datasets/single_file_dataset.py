### single_file_dataset.py
import torch
import os.path
import numpy as np
import pandas as pd
from util import preprocess
from datasets import load_file
from datasets.basic_dataset import BasicDataset

class SingleFileDataset(BasicDataset):
    def __init__(self, param):
        BasicDataset.__init__(self, param)
        self.omics_dims = []

        # Fake A shape inference only
        self.A_dim = 1000  # Placeholder feature count for fake A
        self.sample_list = None

        # Load data for B
        B_df = load_file(param, 'B')
        if param.use_sample_list:
            sample_list_path = os.path.join(param.data_root, 'sample_list.tsv')
            self.sample_list = np.loadtxt(sample_list_path, delimiter='\t', dtype='<U32')
        else:
            self.sample_list = B_df.columns

        if param.use_feature_lists:
            feature_list_B_path = os.path.join(param.data_root, 'feature_list_B.tsv')
            feature_list_B = np.loadtxt(feature_list_B_path, delimiter='\t', dtype='<U32')
        else:
            feature_list_B = B_df.index

        B_df = B_df.loc[feature_list_B, self.sample_list]
        if param.ch_separate:
            B_df_list, self.B_dim = preprocess.separate_B(B_df)
            self.B_tensor_all = []
            for i in range(0, 23):
                B_array = B_df_list[i]
                if self.param.add_channel:
                    B_array = B_array[np.newaxis, :, :]
                B_array = B_array.astype(np.float32)
                B_tensor_part = torch.Tensor(B_array)
                self.B_tensor_all.append(B_tensor_part)
        else:
            self.B_dim = B_df.shape[0]
            B_array = B_df.values
            if self.param.add_channel:
                B_array = B_array[np.newaxis, :, :]
            self.B_tensor_all = torch.Tensor(B_array)

        self.sample_num = B_df.shape[1]
        self.A_tensor_all = torch.zeros((1, self.A_dim, self.sample_num))  # Dummy A tensor
        self.omics_dims.append(self.A_dim)
        self.omics_dims.append(self.B_dim)

    def __getitem__(self, index):
        A_tensor = self.A_tensor_all[:, :, index] if self.param.add_channel else self.A_tensor_all[:, index]
        if self.param.ch_separate:
            B_tensor = [self.B_tensor_all[i][:, :, index] for i in range(23)]
        else:
            B_tensor = self.B_tensor_all[:, :, index] if self.param.add_channel else self.B_tensor_all[:, index]

        return {'A_tensor': A_tensor, 'B_tensor': B_tensor, 'index': index}

    def __len__(self):
        return self.sample_num
