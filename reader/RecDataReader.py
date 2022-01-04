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


#############################################################################
#                       Recommendation Dataset Class                        #
#############################################################################
    
def sample_negative(history, n_candidate, n_neg = -1):
    '''
    Negative sampling for a given user
    @input:
    - history: the users history items
    - n_candidate: number of candidate items to choose from
    - n_neg: number of negative samples for each positive item, -1 if using all candidate items

    @output:
    - negSamples: negative sample matrix of size n_row * n_neg
    '''
    # allow history item as negative sample only when the history size is too large
    if len(history) + n_neg > 0.9 * n_candidate:
        candidates = [i for i in range(1,n_candidate)]
    else:
        candidates = list(set([i for i in range(1,n_candidate)]) - set(history))

    # negative sample matrix of size |items| - |history|
    
    if n_neg == -1:
        neg_samples = candidates
    else:
        neg_samples = np.random.choice(candidates, n_neg, replace=True)
    return neg_samples

def encode_fields(meta_file, fields_meta, fields_vocab):
    '''
    @input:
    - meta_file: meta file name
    - fields_meta: {field_name: {'field_type': 'nominal', 'field_enc': 'v2id'}}
    - field_vocab: {(field_name, value): {'idx': idx}}
    @output:
    - meta: {raw id: {field_name: idx}}
    '''
    df = pd.read_table(meta_file, sep = '\t', engine = 'python')
    id_name = df.columns[0]
    df.index = df[id_name]
    # {idx: {field_name: value}}
    meta = df.to_dict(orient = 'index')
    
    for field_name, field_info in fields_meta.items():
        field_func = eval(field_info['field_enc'])
        for idx, meta_features in meta.items():
            meta[idx][field_name] = field_func(meta_features, fields_vocab, field_name)
    
    meta[0] = {}
    for field_name, field_info in fields_meta.items():
        field_func = eval(field_info['field_enc'])
        meta[0][field_name] = field_func({}, fields_vocab, field_name)
    return meta
    
class RecEvalDataReader(IterableDataset):
    def __init__(self, reader, phase = "val", n_neg = -1, n_worker = 1):
        super().__init__()
        self.reader = reader
        self.phase = phase
        self.n_neg = n_neg
        
        self.n_worker = max(n_worker, 1)
        self.worker_id = None
        
    def __iter__(self):
        for idx in tqdm(range(1, self.reader.n_users)):
            if idx % self.n_worker == self.worker_id:
                yield self.reader.get_user_feed_dict(self.reader.users[idx], self.phase, n_neg = self.n_neg)
                
class RecUserReader(Dataset):
    def __init__(self, reader, phase = "val", n_neg = -1):
        super().__init__()
        self.reader = reader
        self.phase = phase
        self.n_neg = n_neg
    
    def __len__(self):
        return self.reader.n_users - 1
        
    def __getitem__(self, idx):
        return self.reader.get_user_feed_dict(self.reader.users[idx+1], self.phase, n_neg = self.n_neg)
            
class RecDataReader(BaseReader):
    
    @staticmethod
    def parse_data_args(parser):
        '''
        args:
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
        parser = BaseReader.parse_data_args(parser)
        parser.add_argument('--user_meta_data', type = str, required = True,
                            help = 'user meta file path')
        parser.add_argument('--item_meta_data', type = str, required = True,
                            help = 'item meta file path')
        parser.add_argument('--user_fields_meta_file', type = str, required = True,
                            help = 'user field description file path')
        parser.add_argument('--item_fields_meta_file', type = str, required = True,
                            help = 'item field description file path')
        parser.add_argument('--user_fields_vocab_file', type = str, required = True,
                            help = 'user field vocabulary file path')
        parser.add_argument('--item_fields_vocab_file', type = str, required = True,
                            help = 'item field vocabulary file path')
        parser.add_argument('--n_neg', type = int, default = 2, 
                            help = 'number of negative per record for training, set to -1 if sample all items')
        parser.add_argument('--n_neg_val', type = int, default = 200, 
                            help = 'number of negative per user for validation set, set to -1 if sample all items')
        parser.add_argument('--n_neg_test', type = int, default = -1, 
                            help = 'number of negative per user for test set, set to -1 if sample all items')
        return parser
        
    def log(self):
        super().log()
        print(f"\tn_neg : {self.n_neg}")
        print(f"\tn_neg_val : {self.n_neg_val}")
        print(f"\tn_neg_test : {self.n_neg_test}")
        print(f"\tuser_vocab:")
        for k,vMap in self.user_vocab.items():
            print(f"\t\t{k}: L={len(vMap)}")
        print(f"\titem_vocab:")
        for k,vMap in self.item_vocab.items():
            print(f"\t\t{k}: L={len(vMap)}")
        
    def __init__(self, args):
        '''
        - n_neg
        - n_neg_val
        - n_neg_test
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
        '''
        self.n_neg = args.n_neg
        self.n_neg_val = args.n_neg_val
        self.n_neg_test = args.n_neg_test
        super().__init__(args)
        self._build_history()
        self._buffer_negative(args)

    def get_eval_dataset(self):
        if self.phase == "val":
            return RecEvalDataReader(self, phase = self.phase, n_neg = self.n_neg_val, n_worker = self.n_worker)
        elif self.phase == "test":
            return RecEvalDataReader(self, phase = self.phase, n_neg = self.n_neg_test, n_worker = self.n_worker)
        elif self.phase == "train":
            return RecEvalDataReader(self, phase = self.phase, n_neg = self.n_neg, n_worker = self.n_worker)
        
    def _read_data(self, args):
        # read data_file
        super()._read_data(args)
        # load field info
        print(f"Load field meta and vocab file.")
        # dictionary of user field description 
        # e.g. {field_name: {'field_type': 'nominal', 'value_type': int, 'field_enc': 'v2id', 'vocab_key': field_name}}
        self.user_fields = self._load_meta(args.user_fields_meta_file)
        # user field vocabulary for all given field_name in self.user_fields
        # e.g. {field_name, {value: idx}}
        self.user_vocab = self._load_vocab(args.user_fields_vocab_file, self.user_fields)
        print(f"User fields: {list(self.user_fields.keys())}")
        # dictionary of item field description 
        # e.g. {field_name: {'field_type': 'nominal', 'value_type': int, 'field_enc': 'v2id', 'vocab_key': field_name}}
        self.item_fields = self._load_meta(args.item_fields_meta_file)
        # item field vocabulary for all given field_name in self.item_fields
        # e.g. {field_name: {value: idx}}
        self.item_vocab = self._load_vocab(args.item_fields_vocab_file, self.item_fields)
        print(f"Item fields: {list(self.item_fields.keys())}")
        assert "UserID" in self.user_vocab and "ItemID" in self.item_vocab
        # e.g. {UserID: {field_name: value_idx}}
        self.user_meta_file = args.user_meta_data
        self.user_meta = encode_fields(args.user_meta_data, self.user_fields, self.user_vocab)
        # e.g. {ItemID: {field_name: value_idx}}
        self.item_meta = encode_fields(args.item_meta_data, self.item_fields, self.item_vocab)
        
        self.n_users = len(self.user_vocab["UserID"]) + 1  # unseen item has index 0
        self.n_items = len(self.item_vocab["ItemID"]) + 1  # unseen item has index 0
        self.users, self.items = [""] * self.n_users, [""] * self.n_items
        for uid, idx in self.user_vocab["UserID"].items():
            self.users[idx] = uid
        for iid, idx in self.item_vocab["ItemID"].items():
            self.items[idx] = iid
        self.users[0] = "padding"
        self.items[0] = "padding"
    
    def _build_history(self):
        """
        Add history info to data: position
        ! Need data to be sorted by time in ascending order
        """
        print('Appending history info')
        self.user_hist = dict()  # store the already seen sequence of each user
        self.pos_range = {"train": {}, "val": {}, "test": {}}
        for key in ['train', 'val', 'test']:
            df = self.data[key]
            position = list()
            for uid, iid, resp, t in zip(df['UserID'], df['ItemID'], df['Response'], df['Timestamp']):
                if uid not in self.user_hist:
                    self.user_hist[uid] = list()
                pos = len(self.user_hist[uid]) # position in user history
                position.append(pos)
                self.user_hist[uid].append((iid, resp, t))
                if uid not in self.pos_range[key]:
                    self.pos_range[key][uid] = [pos, pos+1]
                else:
                    self.pos_range[key][uid][1] = pos+1
            df['Position'] = position
            # cold-start user may not be in the dataset
            for uid in self.users: 
                if uid not in self.user_hist: # empty history
                    self.user_hist[uid] = list()
                pos = len(self.user_hist[uid])
                if uid not in self.pos_range[key]: # data range include nothing
                    self.pos_range[key][uid] = [pos, pos]
              
    def _buffer_negative(self, args):
        '''
        - bufferred_negative_train_sample: {UserID: [-1,n_neg]}, 
                                            each row is associated with a record in user's history
                                            formulated as a circular list for each user
        - bufferred_negative_val_sample: {UserID: [-1]}, one row associated with each user
        '''
        print("Buffer negative training and validation samples")
        self.bufferred_negative_train_sample = {}
        self.bufferred_negative_val_sample = {}
        for idx in tqdm(range(len(self.users))):
            uid = self.users[idx]
            if len(self.user_hist[uid]) == 0:
                continue
            items, _, __ = zip(*self.user_hist[uid])
            items = [self.get_item_feature(iid, "ItemID") for iid in items]
            start, end = self.pos_range["train"][uid]
            # when training, items in validation/test history is assumed unobserved, 
            # so they have the chance to be considerred as candidates as well
            # 19 is a scalar that determines how many negative samples will be bufferred 
            # and it is just the personal favorite prime
            negitems = np.array(sample_negative(items[:end], self.n_items, 
                                                n_neg = self.n_neg * len(items) * 19)).reshape(-1,self.n_neg)
            self.bufferred_negative_train_sample[uid] = negitems
            # both validation set and test set will consider the entire history as observed.
            # minimum 300 for each user
            if self.n_neg_val > 0:
                self.bufferred_negative_val_sample[uid] = sample_negative(items, self.n_items, n_neg = max(len(items),self.n_neg_val))
                
    def get_item_feature(self, raw_value, field_name):
        if raw_value in self.item_vocab[field_name]:
            return self.item_vocab[field_name][raw_value]
        else:
            return 0
        
    def get_user_feature(self, raw_value, field_name):
        if raw_value in self.user_vocab[field_name]:
            return self.user_vocab[field_name][raw_value]
        else:
            return 0
    
    ###########################
    #        Iterator         #
    ###########################
    
    def __len__(self):
        if self.phase == "train": # recordwise batch during training
            return len(self.data["train"])
        else: # each user as a batch during evaluation
            return self.n_users - 1
        
    def __getitem__(self, idx):
        '''
        train batch after collate:
        {
        'resp': 2, 
        'user_UserID': (B,) 
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
#             neg_items = sample_negative(items, self.n_items, n_neg = self.n_neg)
            negitems = self.bufferred_negative_train_sample[uid]
            # Negative training samples for each user is a cyclic list.
            # Each time the head record is retrieved and put in the back of the list.
            head = negitems[0]
            self.bufferred_negative_train_sample[uid] = np.concatenate((negitems[1:],[head]))
            record = {"resp": resp}
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
        
    def get_user_feed_dict(self, uid, phase, n_neg = -1):
        if len(self.user_hist[uid]) == 0:
            return {"no_item": True}
        items, responses, times = zip(*self.user_hist[uid])
        start, end = self.pos_range[phase][uid]
        if phase == "val":
            if n_neg > 0:
                neg_items = self.bufferred_negative_val_sample[uid]
            else:
                neg_items = sample_negative([self.get_item_feature(iid, "ItemID") for iid in items], 
                                            self.n_items, n_neg = -1)
        elif phase == "test":
            neg_items = sample_negative([self.get_item_feature(iid, "ItemID") for iid in items], self.n_items, n_neg = n_neg)
        elif phase == "train":
            u_neg_items = self.bufferred_negative_train_sample[uid]
            neg_items = u_neg_items[:end]
            self.bufferred_negative_train_sample[uid] = np.concatenate((u_neg_items[end:],neg_items))
            neg_items = np.array(neg_items).reshape(-1)
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
    
    def get_user_meta(self, uid):
        if uid in self.user_meta:
            return self.user_meta[uid]
        else:
            return self.user_meta[0]
    
    def get_item_meta(self, iid, from_idx = False):
        iid = self.items[iid] if from_idx else iid
        if iid in self.item_meta:
            return self.item_meta[iid]
        else:
            return self.item_meta[0]
        
    def get_item_list_meta(self, iid_list, from_idx = False):
        '''
        @input:
        - iid_list: item id list
        @output:
        - meta_data: {field_name: (B,feature_size)}
        '''
        meta_data = [self.get_item_meta(iid, from_idx) for iid in iid_list]
        return {k: list(itertools.chain([iid_meta[k] for iid_meta in meta_data])) for k in meta_data[0]}

    def get_average_response(self):
        return self.data["train"]["Response"].mean()
    
    def get_statistics(self):
        '''
        - n_user
        - n_item
        - s_parsity
        - from BaseReader:
            - length
            - fields
        '''
        stats = super().get_statistics()
        # number of users
        stats["n_user"] = self.n_users
        # number of items
        stats["n_item"] = self.n_items
        stats["sparsity"] = float(stats["length"]) / (self.n_users * self.n_items)
        # average response
        stats["avg_resp"] = self.get_average_response()
        return stats
    
