import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
 
from model.components import init_weights
from model.base_rec import GeneralRecModel
            
class FederatedRecModel(GeneralRecModel):
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - device_dropout_p
        - device_dropout_type
        - n_local_step
        - random_local_step
        - from GeneralRecModel:
            - emb_size
            - from BaseModel:
                - model_path
                - loss
                - l2_coef
                - cuda
                - optimizer
                - n_worker
        '''
        parser = GeneralRecModel.parse_model_args(parser)
        parser.add_argument('--device_dropout_p', type=float, default=0.5, 
                            help='the possibility of device dropout')
        parser.add_argument('--device_dropout_type', type=str, default='same',
                            help='[same, max]')
        parser.add_argument('--n_local_step', type=int, default=1,
                            help='the number of local optimization steps between synchronization')
        parser.add_argument('--random_local_step', action='store_true',
                            help='True if adopts random number of local step')
        parser.add_argument('--aggregation_func', type=str, default='fedavg',
                            help='[fedavg, fedprox]')
        parser.add_argument('--mitigation_trade_off', type=float, default=0.5,
                            help='trade-off factor between old param and param proposals, new = alpha * proposal + (1-alpha) * old')
        parser.add_argument('--elastic_mu', type=float, default=0.01,
                            help='trade-off factor local objective and elastic objective, grad = obj_grad + mu * elastic_grad')
        return parser
    
    @staticmethod
    def get_reader():
        return "FedRecReader"
    
    def log(self):
        super().log()
        print("\tdevice_dropout_p = " + str(self.device_dropout_p))
        print("\tdevice_dropout_type = " + str(self.device_dropout_type))
        print("\tn_local_step = " + str(self.n_local_step))
        print("\trandom_local_step = " + str(self.random_local_step))
        print("\tmitigation_trade_off = " + str(self.mitigation_beta))
        print("\taggregation_func = " + str(self.aggregation_func))
        print("\telastic_mu = " + str(self.elastic_mu))
        
    def __init__(self, args, reader, device):
        self.device_dropout_p = args.device_dropout_p
        self.device_dropout_type = args.device_dropout_type
        self.n_local_step = args.n_local_step
        self.random_local_step = args.random_local_step
        self.mitigation_beta = args.mitigation_trade_off
        self.n_device = reader.n_users
        self.aggregation_func = args.aggregation_func
        self.elastic_mu = args.elastic_mu
        super().__init__(args, reader, device)
        if self.device_dropout_type == "max":
            self.individual_p = np.random.uniform(0.,self.device_dropout_p,self.n_device)
            self.individual_p[0] = self.device_dropout_p
        
#     def get_regularization(self, *modules):
# #         p = modules[0].parameters()
# #         reg = torch.mean(p * p)
#         reg = 0
#         for m in modules:
#             for p in m.parameters():
#                 reg = torch.sum(p * p) + reg
#         return reg
            
#     def get_loss(self, feed_dict: dict, out_dict: dict):
#         """
#         The same as GeneralRecModel.get_loss(), except changing the loss terms from means to sum
#         """
        
#         preds, reg = out_dict["preds"].view(-1), out_dict["reg"] # size (B,)

#         # loss
#         if self.loss_type == "regression":
#             resps = feed_dict["resp"] # size (B,)
#             loss = torch.mean(self.mse_loss(preds, resps.view(-1))) + self.l2_coef * reg
#             return loss
#         else:
#             B = preds.shape[0]
#             neg_preds = out_dict["neg_preds"].view(-1) # size (B*L,)
#             neg_reg = out_dict["neg_reg"] # scalar
#             ratio = len(neg_preds) / len(preds) # number of negative samples per positive sample
#             extended_preds = preds.view(-1,1).repeat(1,int(ratio)).view(-1)
#             if self.loss_type == "pairwisebpr":
#                 loss = torch.sum(self.sigmoid(neg_preds - extended_preds))
#             elif self.loss_type == "pairwisemrl":
#                 y = torch.ones_like(neg_preds)
#                 mrl = self.mrl_loss(self.sigmoid(extended_preds), self.sigmoid(neg_preds), y)
#                 loss = torch.sum(mrl)
#             elif self.loss_type == "pointwise":
#                 pos_target = torch.ones_like(extended_preds)
#                 neg_target = torch.zeros_like(neg_preds)
#                 loss = torch.sum(self.bce_loss(self.sigmoid(extended_preds), pos_target)) + \
#                         torch.sum(self.bce_loss(self.sigmoid(neg_preds), neg_target))
#             elif self.loss_type == "softmax":
#                 labels = torch.tensor([0] * B).to(self.device)
#                 combined_preds = torch.cat([preds.view(B,1),neg_preds.view(B,-1)], dim = 1)
#                 loss = torch.sum(self.ce_loss(combined_preds, labels))
#             else:
#                 raise NotImplemented
#             loss = loss + self.l2_coef * (reg * ratio + neg_reg)
#             return loss
        
    ##################################
    #        federated control       #
    ##################################
    
    '''
    General federated learning with central node:
    model.actions_before_train() --> model.keep_cloud_params()
    for each epoch:
        model.actions_before_epoch() --> Set up proposal buffer for later aggregation
        for each user_batch:
            local_info = model.get_local_info(user_batch, {some training info})
            if not model.do_device_dropout():
                model.download_cloud_params()
                model.local_optimize()
                model.upload_edge_params()
        model.actions_after_epoch() --> model.mitigate_params()
    '''
    
    def do_device_dropout(self, info_dict):
        device_id = info_dict['device']
        if device_id < 0 or device_id >= self.n_device:
            device_id = 0
        p = self.device_dropout_p if self.device_dropout_type == "same" else self.individual_p[device_id]
        return np.random.random() < p
        
    def keep_cloud_params(self):
        '''
        Keep a copy of current parameters as cloud copy 
        All users will synchronize from this copy later using download_cloud_params()
        It also reset user proposal and their counter
        '''
        self.cloud_params = {}
        for name, param in self.named_parameters():
            self.cloud_params[name] = param.data.clone()
            print(f"\t{name}:\n{self.cloud_params[name]}")
        
    def download_cloud_params(self, local_info):
        '''
        Copy cloud parameters into local parameters
        '''
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.data = self.cloud_params[name].clone()
    
    def get_local_info(self, feed_dict, training_info):
        '''
        Identify what information is observed in the local space.
        @input:
        - feed_dict: user-wise batch data
        - training_info: {"lr": local learning rate, "epoch": epoch_id}
        '''
        training_info['device'] = feed_dict["user_UserID"].reshape(-1).detach().cpu().numpy()[0]
        training_info["user"] = feed_dict['user_UserID'].reshape(-1).detach()
        training_info["item"] = feed_dict['item_ItemID'].reshape(-1).detach()
        training_info["neg_item"] = feed_dict['negi_ItemID'].reshape(-1).detach()
        return training_info
                
        
    def mitigate_params(self):
        '''
        Mitigate parameters and update cloud parameters
        Final mapping is the convex combination: (1 - alpha) * old_params + alpha * average_proposal
        '''
#         with torch.no_grad():
#             for name, param in self.named_parameters():
#                 if name in self.cloud_params and name in self.param_proposal:
#                     self.cloud_params[name] = self.cloud_params[name] * (1-self.mitigation_beta) + \
#                                             self.param_proposal[name] * self.mitigation_beta / self.param_proposal_count[name]
        with torch.no_grad():
            for name, param in self.cloud_params.items():
                if name in self.param_proposal:
#                     agg = self.cloud_params[name] * (1-self.mitigation_beta) + \
#                             self.param_proposal[name] * self.mitigation_beta / self.param_proposal_count[name]#.view(-1,1)
                    agg = self.param_proposal[name] / self.param_proposal_count[name]#.view(-1,1)
                    select = self.param_proposal_count[name].view(-1) > 0
                    self.cloud_params[name][select] = agg[select]
    
    def actions_before_train(self, info):
        self.keep_cloud_params() # store parameter to cloud space
        
    def actions_after_epoch(self, info):
        self.mitigate_params() # mitigate proposals
    
    def local_optimize(self, feed_dict, local_info):
        '''
        Model optimization in local space
        '''
#         device_id = info_dict['device']
#         if device_id < 0 or device_id >= self.n_device:
#             device_id = 0
        learning_rate = local_info['lr']

        # get number of local rounds
        local_rounds = self.n_local_step if not self.random_local_step else np.random.randint(1,max(self.n_local_step+1,2))
        
    
        # local update
        local_loss = []
        for i in range(local_rounds):
            local_info['local_step'] = i
            # local optimization
            self.optimizer.zero_grad()
            out_dict = self.forward(feed_dict)
            loss = self.get_loss(feed_dict, out_dict)
            loss.backward()
            self.local_gradient_manipulation(local_info)
            # apply gradient to local parameters before upload
            if self.aggregation_func == "fedavg":
                for name, param in self.named_parameters():
                    param.data -= learning_rate * param.grad
            elif self.aggregation_func == "fedprox":        
                for name, param in self.named_parameters():
                    # fedprox objective: local_obj + 0.5 * mu * |w - w(t)|^2
                    # fedprox grad: local_grad + mu * (w - w(t))
                    param.data -= learning_rate * (param.grad * (1+self.elastic_mu) - self.elastic_mu * self.cloud_params[name].data)
            elif self.aggregation_func == "shared":
                self.optimizer.step()
            local_loss.append(loss.item())
            
        if self.loss_type == "regression":
            raise NotImplemented
        else:
            return {"loss": local_loss[-1]}
        
    def local_gradient_manipulation(self, local_info):
        '''
        Optional local operations after the gradient is calculated before upload.
        '''
        pass
        
    def actions_before_epoch(self, info):
        '''
        Set up proposal for later aggregation
        '''
#         with torch.no_grad():
#             for name, param in self.cloud_params.items():
#                 print(name)
#                 print(param)
    
    def upload_edge_params(self, local_info):
        '''
        Upload edge parameters to cloud
        '''
        pass
#         with torch.no_grad():
#             for name, param in self.named_parameters():
#                 self.param_proposal[name] += param.data.clone()
#                 self.param_proposal_count[name] += 1