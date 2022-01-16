import numpy as np
import torch
import torch.nn as nn

from model.fed_rec.general_fed_rec import FederatedRecModel
from model.components import FieldEmbeddings, init_weights

class FedMF(FederatedRecModel):
        
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
        self.local_info = []

    def get_forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {"UserID": (B,), "ItemID": (B,N)}
        @output:
        - result_dict: {"preds": (B,N), "reg": scalar}
        
        '''
        user_ids = feed_dict['UserID'] 
        item_ids = feed_dict['ItemID']
        B = item_ids.shape[0]
        item_ids = item_ids.view(B,-1)
        L = item_ids.shape[1]
        user_ids = user_ids.view(-1,1)

        # get embeddings
        user_emb = self.uEmb(user_ids)
        item_emb = self.iEmb(item_ids)
        # get user and item bias
        user_bias = self.uBias(user_ids)
        item_bias = self.iBias(item_ids)
        # prediction
        prediction = torch.sum(user_emb * item_emb, dim = -1).view(B,L,1)
        prediction = prediction + item_bias + user_bias
        prediction = prediction + self.global_bias
        # regularization terms
        reg = torch.mean(user_emb * user_emb) + torch.mean(item_emb * item_emb) + \
                torch.mean(user_bias * user_bias) + torch.mean(item_bias * item_bias)
        return {'preds': prediction.view(B,L), "reg": reg}
    
#     def keep_cloud_params(self):
#         super().keep_cloud_params()

#     def download_cloud_params(self, local_info):
#         super().download_cloud_params(local_info)

#     def mitigate_params(self):
#         super().mitigate_params()
        
    def upload_edge_params(self, local_info):
        '''
        Upload edge parameters to cloud
        '''
        with torch.no_grad():
            self.param_proposal['global_bias'] += self.global_bias.data.clone()
            self.param_proposal_count['global_bias'] += 1
            # user parameters
            user_ids = local_info['user'].view(-1)
#             print("uEmb.weight")
#             print(self.param_proposal["uEmb.weight"][user_ids])
#             print(self.cloud_params["uEmb.weight"][user_ids])
            self.param_proposal["uEmb.weight"][user_ids] += self.uEmb.weight.data[user_ids].clone()
#             print(self.param_proposal["uEmb.weight"][user_ids])
            self.param_proposal["uBias.weight"][user_ids] += self.uBias.weight.data[user_ids].clone()
            self.param_proposal_count["uEmb.weight"][user_ids] += 1
            self.param_proposal_count["uBias.weight"][user_ids] += 1
            # item parameters
            for key in local_info:
#                 print(key)
                if "item" in key:
                    item_ids = local_info[key].view(-1)
#                     print(self.param_proposal["iEmb.weight"][item_ids])
#                     print(self.iEmb.weight.data[item_ids])
#                     print(self.cloud_params["iEmb.weight"][item_ids])
                    self.param_proposal["iEmb.weight"][item_ids] += self.iEmb.weight.data[item_ids].clone()
#                     print(self.param_proposal["iEmb.weight"][item_ids])
                    self.param_proposal["iBias.weight"][item_ids] += self.iBias.weight.data[item_ids].clone()
                    self.param_proposal_count["iEmb.weight"][item_ids] += 1
                    self.param_proposal_count["iBias.weight"][item_ids] += 1
#                 input()
        self.local_info = []
        
    def mitigate_params(self):
        with torch.no_grad():
            for name, param in self.cloud_params.items():
#                 print(name)
#                 print(self.cloud_params[name])
                if name == 'global_bias':
                    sum_grad = self.param_proposal[name] - self.cloud_params[name] * self.param_proposal_count[name].view(-1)
                    self.cloud_params[name] = self.cloud_params[name] + self.mitigation_beta * sum_grad
#                     self.cloud_params[name] = self.param_proposal[name] / self.param_proposal_count[name].view(-1)
                elif name in self.param_proposal:
#                     agg = self.cloud_params[name] * (1-self.mitigation_alpha) + \
#                             self.param_proposal[name] * self.mitigation_alpha / self.param_proposal_count[name].view(-1,1)
#                     agg = self.param_proposal[name] / self.param_proposal_count[name].view(-1,1)
                    sum_grad = self.param_proposal[name] - self.cloud_params[name] * self.param_proposal_count[name].view(-1,1)
#                     print(self.param_proposal[name])
#                     print(sum_grad)
                    self.cloud_params[name] = self.cloud_params[name] + self.mitigation_beta * sum_grad
#                     print(agg)
#                     select = self.param_proposal_count[name] > 0
#                     self.cloud_params[name][select] = agg[select]
#                     print(select)
#                     input()
#                 print(self.cloud_params[name])
#                 print(self.param_proposal_count[name])
#                 input()
        
    def actions_before_epoch(self, info):
        super().actions_before_epoch(info)
        self.param_proposal = {k: torch.zeros_like(v) for k,v in self.cloud_params.items()}
        self.param_proposal_count = {k: torch.zeros(v.shape[0]).to(torch.long).to(self.device) for k,v in self.cloud_params.items()}
        
    