import numpy as np
import torch
import torch.nn as nn

from model.base_rec import GeneralRecModel 
from model.components import FieldEmbeddings, init_weights

class MF(GeneralRecModel):
        
    def _define_params(self, args, reader):
        super()._define_params(args, reader)
        self.uEmb = nn.Embedding(reader.n_users, self.emb_size, padding_idx=0)
        self.iEmb = nn.Embedding(reader.n_items, self.emb_size, padding_idx=0)
        self.uBias = nn.Embedding(reader.n_users, 1, padding_idx=0)
        self.iBias = nn.Embedding(reader.n_items, 1, padding_idx=0)
        for module in [self.uEmb, self.iEmb]:
            init_weights(module)
        for module in [self.uBias, self.iBias]:
            module.weight.data *= 0

    def get_forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {"UserID": (B,), "ItemID": (B,N)}
        @output:
        - result_dict: {"preds": (B,N), "reg": scalar}
        
        '''
#         self.check_list = []
#         B = feed_dict["batch_size"]
        user_ids = feed_dict['UserID'] 
        item_ids = feed_dict['ItemID']
        B = item_ids.shape[0]
        item_ids = item_ids.view(B,-1) # (B,1) or (B,L)
        L = item_ids.shape[1]
        user_ids = user_ids.view(-1,1) # (B,1) or (1,1)

        # get embeddings
        user_emb = self.uEmb(user_ids) # (B,1,d) or (1,1,d)
        item_emb = self.iEmb(item_ids) # (B,1,d) or (B,L,d)
        # get user and item bias
        user_bias = self.uBias(user_ids) # (B,1,1) or (1,1,1)
        item_bias = self.iBias(item_ids) # (B,1,1) or (B,L,1)
        # prediction
        # (B,1,d)*(B,1,d) --> (B,1,d), (B,1,d)*(B,L,d) --> (B,L,d), (1,1,d)*(B,L,d) --> (B,L,d)
        prediction = torch.sum(user_emb * item_emb, dim = -1).view(B,L,1)
        prediction = prediction + item_bias + user_bias
        prediction = prediction + self.global_bias
        # regularization terms
        reg = torch.mean(user_emb * user_emb) + torch.mean(item_emb * item_emb) + \
                torch.mean(user_bias * user_bias) + torch.mean(item_bias * item_bias)
        return {'preds': prediction.view(B,L),  "reg": reg}
    