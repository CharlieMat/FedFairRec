import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from tqdm import tqdm

from reader.reader_utils import *
    
def worker_init_func(worker_id):
    worker_info = data.get_worker_info()
    worker_info.dataset.worker_id = worker_id
    
#############################################################################
#                              Dataset Class                                #
#############################################################################

class BaseReader(Dataset):
    
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--data_file', type=str, required=True, 
                            help='data_file_path')
#         parser.add_argument('--data_fields_meta_file', type=str,
#                             help='data fields definition file')
#         parser.add_argument('--data_fields_vocab_file', type=str,
#                             help='data fields vocab file')
        parser.add_argument('--n_worker', type=int, default=4,
                            help='number of worker for dataset loader')
        return parser
    
    @staticmethod
    def _load_meta(meta_file):
        '''
        @output:
        - meta: {name: {field: value}}
        '''
        df = pd.read_table(meta_file, sep = "\t", engine = 'python', index_col = 0)
        return df.to_dict(orient = 'index')
    
    @staticmethod
    def _load_vocab(vocab_file, meta):
        '''
        @input:
        - vocab_file: vocab file path
        - meta: field meta data, {field_name: {'field_type': 'nominal', 
                                                'value_type': int, 
                                                'field_enc': 'v2id'}}
        @output:
        - vocab: {field_name: {value: idx}}
        '''
        print("Load vocab from : " + vocab_file)
        item_vocab = pd.read_table(vocab_file, index_col = 1) # value as index
        vocab = {}
        # extract vocab from dataframe
        for f in item_vocab['field_name'].unique():
            type_func = eval(meta[f]['value_type'])
            if f in meta:
                # {value: {'idx': value_id}}
                value_idx = item_vocab[item_vocab['field_name'] == f][['idx']]
                value_idx = value_idx[~value_idx.index.duplicated(keep='first')].to_dict(orient = 'index')
                # {value: value_id}
                vocab[f] = {type_func(k): vMap['idx'] for k,vMap in value_idx.items()}
        return vocab
    
    def log(self):
        print("Reader params:")
        print(f"\tn_worker: {self.n_worker}")
        for k,v in self.get_statistics().items():
            print(f"\t{k}: {v}")
            
    def __init__(self, args):
        '''
        - phase: one of ["train", "val", "test"]
        - data: {phase: pd.DataFrame}
        - data_fields: {field_name: (field_type, field_var)}
        - data_vocab: {field_name: {value: index}}
        '''
        self.phase = "train"
        self.n_worker = args.n_worker
        self._read_data(args)
        
    def _read_data(self, args):
        self.data = dict()
        for phase in ["train", "val", "test"]:
            print(f"Loading {phase} data file", end = '\r')
            self.data[phase] = pd.read_table(args.data_file + phase + ".tsv", 
                                         sep = '\t', engine = 'python')
            print(f"Loading {phase} data file. Done.")
        print(self.data["train"].head())
#         self.data = dict()
#         if args.data_fields_meta_file:
#             # select certain fields
#             self.data_fields = self._load_meta(args.data_fields_meta_file)
#             self.data_vocab = self._load_vocab(args.data_fields_vocab_file)
#             for k in ["train", "val", "test"]:
#                 print(f"Loading {k} data file", end = '\r')
#                 self.data[k] = pd.read_table(args.data_file + k + ".tsv", 
#                                              sep = '\t', engine = 'python',
#                                              usecols=self.data_fields.keys())
#                 print(f"Loading {k} data file. Done.")
#         else:
#             for k in ["train", "val", "test"]:
#                 print(f"Loading {k} data file", end = '\r')
#                 self.data[k] = pd.read_table(args.data_file + k + ".tsv", 
#                                              sep = '\t', engine = 'python')
#                 print(f"Loading {k} data file. Done.")
#             self.data_fields = {f:{} for f in self.data["train"].columns}
#             print(self.data["train"].head())

    def get_statistics(self):
        return {'length': len(self)}
    
    def set_phase(self, phase):
        assert phase in ["train", "val", "test"]
        self.phase = phase
        
    def get_train_dataset(self):
        return self
    
    def get_eval_dataset(self):
        return self
    
    def __len__(self):
        return len(self.data[self.phase])
    
    def __getitem__(self, idx):
        pass
    

    
    