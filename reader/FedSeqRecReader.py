import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm
import itertools

from reader.BaseReader import BaseReader
from reader.reader_utils import *
from reader.FedRecReader import FedRecReader, FedRecIterator

#############################################################################
#                       Recommendation Dataset Class                        #
#############################################################################
            
class FedSeqRecReader(FedRecReader):
    
    @staticmethod
    def parse_data_args(parser):
        '''
        - max_seq_length
        - from FedRecReader:
            - user_meta_data
            - item_meta_data
            - user_fields_meta_file
            - item_fields_meta_file
            - user_fields_vocab_file
            - item_fields_vocab_file
            - n_neg
            - n_neg_val
            - n_neg_test
            - from BaseReader:
                - data_file
                - n_worker
        '''
        parser = FedRecReader.parse_data_args(parser)
        parser.add_argument('--max_seq_length', type = int, default = 15,
                            help = 'user meta file path')
        return parser

    def log(self):
        super().log()
        print(f"\tmax_seq_length : {self.max_seq_length}")
        
    def __init__(self, args):
        self.max_seq_length = args.max_seq_length
        super().__init__(args)
    
    def get_user_feed_dict(self, uid, phase, n_neg = -1):
        if len(self.user_hist[uid]) == 0:
            return {"no_item": True}
        items, responses, times = zip(*self.user_hist[uid])
        start, end = self.pos_range[phase][uid]
        
        if phase == "val":
            neg_items = self.bufferred_negative_val_sample[uid]
            sequence = self.get_item_list_meta(padding_and_cut(list(items[:start]), self.max_seq_length))["ItemID"]
            seq_length = min(start, self.max_seq_length)
        elif phase == "test":
            neg_items = sample_negative([self.get_item_feature(iid, "ItemID") for iid in items], self.n_items, n_neg = n_neg)
            sequence = self.get_item_list_meta(padding_and_cut(list(items[:start]), self.max_seq_length))["ItemID"]
            seq_length = min(start, self.max_seq_length)
        elif phase == "train":
            negitems = self.bufferred_negative_train_sample[uid]
            head = negitems[:end]
            self.bufferred_negative_train_sample[uid] = np.concatenate((negitems[end:],head))
            neg_items = [self.get_item_feature(iid, "ItemID") for iid in head.reshape(-1)]
            sequence = [self.get_item_list_meta(padding_and_cut(list(items[:i]), self.max_seq_length))["ItemID"] for i in range(start,end)]
            seq_length = np.array([min(i, self.max_seq_length) for i in range(start,end)])
        else:
            raise NotImplemented
        items = items[start:end]
        user_data = {"resp": np.array(responses[start:end]), 
                     "user_Sequence": np.array(sequence), 
                     "user_SeqLen": seq_length}
        for k,v in self.get_user_meta(uid).items():
            user_data["user_" + k] = np.array(v)
        if len(items) > 0:
            for k,v in self.get_item_list_meta(items).items():
                user_data["item_" + k] = np.array(v)
            for k,v in self.get_item_list_meta(neg_items, from_idx = True).items():
                user_data["negi_" + k] = np.array(v)
        else:
            user_data["no_item"] = True
        return user_data
    
    ###########################
    #        Iterator         #
    ###########################
    
    def __len__(self):
        return self.n_users - 1
        
    def __getitem__(self, idx):
        raise NotImplemented
        
    