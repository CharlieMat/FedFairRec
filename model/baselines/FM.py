import numpy as np
import torch
import torch.nn as nn

from model.base_rec import GeneralRecModel 
from model.components import FieldEmbeddings, init_weights

class FM(GeneralRecModel):
    @staticmethod
    def parse_model_args(parser):
        return GeneralRecModel.parse_model_args(parser)
        
    def _define_params(self, args, reader):
        super()._define_params(args, reader)
        self.user_embs = FieldEmbeddings(reader.user_fields, reader.user_vocab, args.emb_size, combiner_type = "sum")
        self.item_embs = FieldEmbeddings(reader.item_fields, reader.item_vocab, args.emb_size, combiner_type = "sum")
        self.u_linear = nn.Linear(len(reader.user_fields),1)
        self.i_linear = nn.Linear(len(reader.item_fields),1)
        self.second_dense = nn.Linear(args.emb_size, 1)
        init_weights(self.u_linear)
        init_weights(self.i_linear)
        init_weights(self.second_dense)

    def get_forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {"UserID": (B,1), "ItemID": (B,1)} or {"UserID": (B,1), "ItemID": (B,N)} or {"UserID": (1,1), "ItemID": (B,1)}
        @output:
        - result_dict: {"preds": (B,N), "reg": scalar}, where N >= 1
        
        '''
        user_ids = feed_dict['UserID'] 
        item_ids = feed_dict['ItemID']
        B = item_ids.shape[0]
        item_ids = item_ids.view(B,-1) # (B,1) or (B,N)
        N = item_ids.shape[1]
        user_ids = user_ids.view(-1,1) # (B,1) or (1,1)
        
        feed_dict["N"] = 1
        u_emb_dict = self.user_embs(feed_dict)
        feed_dict["N"] = N
        i_emb_dict = self.item_embs(feed_dict)
        # (B,1,K) or (1,1,K)
        u_scalar = torch.cat([v for v in u_emb_dict["scalar"].values()], dim = 2)
        # (B,1,K,d) or (1,1,K,d)
        u_vector = torch.cat([v for v in u_emb_dict["vector"].values()], dim = 2)
        # (B,N,K)
        i_scalar = torch.cat([v for v in i_emb_dict["scalar"].values()], dim = 2)
        # (B,N,K,d)
        i_vector = torch.cat([v for v in i_emb_dict["vector"].values()], dim = 2)
        # sum up the scalars as linear output, (B,N)
        linear_part = self.u_linear(u_scalar).view(-1,1) + self.i_linear(i_scalar).view(B,N)
        # (B,N,d)
        square_of_sum = torch.pow(torch.sum(u_vector, dim = 2) + torch.sum(i_vector, dim = 2), 2)
        # (B,N,d)
        sum_of_square = torch.sum(torch.pow(u_vector, 2), dim = 2) + torch.sum(torch.pow(i_vector, 2), dim = 2)
        # second order term = (x1 + x2 + ... xn)^2 - (x1^2 + x2^2 + ... + xn^2)
        second_order_part = self.second_dense(0.5 * (square_of_sum - sum_of_square)).view(B,N)
        # (B,N)
        prediction = linear_part + second_order_part
        prediction = prediction + self.global_bias
        
        reg = self.get_regularization(self.u_linear, self.i_linear, self.second_dense)
        reg += torch.mean(u_vector * u_vector) + torch.mean(i_vector * i_vector) + \
                torch.mean(u_scalar * u_scalar) + torch.mean(i_scalar * i_scalar)
        
        return {'preds': prediction,  "reg": reg}
    