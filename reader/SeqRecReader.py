import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import itertools

from reader.reader_utils import *
from reader.RecDataReader import sample_negative, encode_fields, RecDataReader
    
class SeqRecReader(RecDataReader):
    
    @staticmethod
    def parse_data_args(parser):
        '''
        args:
        - max_seq_length
        - from RecDataReader
            - user_meta_file
            - item_meta_file
            - user_fields_meta_file
            - item_fields_meta_file
            - user_fields_vocab_file
            - item_fields_vocab_file
            - n_neg
            - n_neg_val
            - n_neg_test
            - from BaseReader:
                - data_file
        '''
        parser = RecDataReader.parse_data_args(parser)
        parser.add_argument('--max_seq_length', type = int, default = 15,
                            help = 'user meta file path')
        return parser
        
    def log(self):
        super().log()
        print(f"\tmax_seq_length : {self.max_seq_length}")
        
    def __init__(self, args):
        '''
        - max_seq_length
        - from RecDataReader
            - n_neg
            - user_meta: {user id: {field_name: idx}}
            - user_fields: {field_name: (field_type, field_var)}
            - user_vocab: {field_name: {value: index}}
            - item_meta: {item id: {field_name: idx}}
            - item_fields: {field_name: (field_type, field_var)}
            - item_vocab: {field_name: {value: index}}
            - n_users
            - n_items
            - users: [user id]
            - user_hist: {user id: [(item id, response, time)]}
            - pos_range: {phase: {user id: [pos_start, pos_end]}}
            - from BaseReader:
                - phase
                - data: will add Position column
                - data_vocab
        '''
        self.max_seq_length = args.max_seq_length
        super().__init__(args)
                    
    def get_user_feed_dict(self, uid, phase, n_neg = -1):
        if len(self.user_hist[uid]) == 0:
            return {"no_item": True}
        items, responses, times = zip(*self.user_hist[uid])
        start, end = self.pos_range[phase][uid]
        sequence = self.get_item_list_meta(padding_and_cut(list(items[:start]), self.max_seq_length))["ItemID"]
        if phase == "val":
            neg_items = self.bufferred_negative_val_sample[uid]
        elif phase == "test":
            neg_items = sample_negative(items, self.n_items, n_neg = n_neg)
        elif phase == "train":
            neg_items = self.bufferred_negative_train_sample[uid].reshape(-1)
        else:
            print(uid,phase)
            raise NotImplemented
        items = items[start:end]
        user_data = {"resp": np.array(responses[start:end]), 
                     "user_Sequence": np.array(sequence), 
                     "user_SeqLen": min(start, self.max_seq_length)}
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
        
    def __getitem__(self, idx):
        '''
        train batch after collate:
        {
        'resp': 2, 
        'user_UserID': (B,) 
        'user_Sequence': (B,|H|)
        'user_SeqLen': (B,)
        'user_XXX': (B,feature_size)
        'item_ItemID': (B,)
        'item_XXX': (B,feature_size)
        'negi_ItemID': (B,n_neg) 
        'negi_XXX': (B,n_neg,feature_size) 
        }
        '''
        
        if self.phase != "train":
            raise NotImplemented
        else:
            uid, iid, resp, t, pos = self.data["train"].iloc[idx]
            items, _, __ = zip(*self.user_hist[uid])
            neg_items = self.bufferred_negative_train_sample[uid]
            # Negative training samples for each user is a cyclic list.
            # Each time the head record is retrieved and put in the back of the list.
            head = neg_items[0]
            self.bufferred_negative_train_sample[uid] = np.concatenate((neg_items[1:],[head]))
            sequence = self.get_item_list_meta(padding_and_cut(list(items[:pos]), self.max_seq_length))["ItemID"]
            record = {"resp": np.array(resp), 
                      "user_Sequence": np.array(sequence), 
                      "user_SeqLen": min(pos, self.max_seq_length)}
            for k,v in self.get_user_meta(uid).items():
                record["user_" + k] = np.array(v)
            if len(items) > 0:
                for k,v in self.get_item_meta(iid).items():
                    record["item_" + k] = np.array(v)
                for k,v in self.get_item_list_meta(head, from_idx = True).items():
                    record["negi_" + k] = np.array(v)
            else:
                record["no_item"] = True
            return record
    