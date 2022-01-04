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
from reader.RecDataReader import sample_negative, encode_fields, RecDataReader

#############################################################################
#                       Recommendation Dataset Class                        #
#############################################################################
    
class FedRecIterator(IterableDataset):
    def __init__(self, reader, phase = "val", n_neg = -1, n_worker = 1):
        super().__init__()
        self.reader = reader
        self.phase = phase
        self.n_neg = n_neg
        
        self.n_worker = max(n_worker, 1)
        self.worker_id = None
        
    def __iter__(self):
        for idx in tqdm(range(1, self.reader.n_users)): # skip padding user
            if idx % self.n_worker == self.worker_id:
                yield self.reader.get_user_feed_dict(self.reader.users[idx], self.phase, n_neg = self.n_neg)
            
class FedRecReader(RecDataReader):
    
    @staticmethod
    def parse_data_args(parser):
        '''
        from RecDataReader:
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
        parser = RecDataReader.parse_data_args(parser)
        return parser

    def get_train_dataset(self):
        return FedRecIterator(self, phase = "train", 
                              n_neg = self.n_neg, 
                              n_worker = self.n_worker)

    def get_eval_dataset(self):
        return FedRecIterator(self, phase = self.phase, 
                              n_neg = self.n_neg_val if self.phase == "val" else self.n_neg_test, 
                              n_worker = self.n_worker)
        
#     def _buffer_negative(self, args):
#         '''
#         - bufferred_negative_train_sample: {UserID: [n_neg]}, one row associated with each user
#         - bufferred_negative_val_sample: {UserID: [n_neg_val]}, one row associated with each user
#         '''
#         print("Buffer negative training and validation samples")
#         self.bufferred_negative_train_sample = {}
#         self.bufferred_negative_val_sample = {}
#         for idx in tqdm(range(len(self.users))):
#             uid = self.users[idx]
#             if len(self.user_hist[uid]) == 0:
#                 continue
#             items, _, __ = zip(*self.user_hist[uid])
#             items = [self.get_item_feature(iid, "ItemID") for iid in items]
#             start, end = self.pos_range["train"][uid]
#             # when training, items in validation/test history is assumed unobserved, 
#             # so they have the chance to be considerred as candidates as well
#             self.bufferred_negative_train_sample[uid] = sample_negative(items[:end], self.n_items, 
#                                                                         n_neg = max(len(items),self.n_neg))
#             # both validation set and test set will consider the entire history as observed.
#             self.bufferred_negative_val_sample[uid] = sample_negative(items, self.n_items, 
#                                                                       n_neg = max(len(items),self.n_neg_val))

    def get_user_feed_dict(self, uid, phase, n_neg = -1):
        if len(self.user_hist[uid]) == 0:
            return {"no_item": True}
        items, responses, times = zip(*self.user_hist[uid])
        start, end = self.pos_range[phase][uid]
        if phase == "val":
            neg_items = self.bufferred_negative_val_sample[uid]
        elif phase == "test":
            neg_items = sample_negative([self.get_item_feature(iid, "ItemID") for iid in items], self.n_items, n_neg = n_neg)
        elif phase == "train":
            negitems = self.bufferred_negative_train_sample[uid]
            head = negitems[:end]
            self.bufferred_negative_train_sample[uid] = np.concatenate((negitems[end:],head))
            neg_items = [self.get_item_feature(iid, "ItemID") for iid in head.reshape(-1)]
        else:
            raise NotImplemented
        items = items[start:end]
        user_data = {"resp": np.array(responses[start:end])}
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
        
    