import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
 
from model.components import init_weights
from model.fed_rec.general_fed_rec import FederatedRecModel
from model.fair_rec.Fed_FUGP import Fed_FUGP
            
class FedFairRecModel(FederatedRecModel):
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - from FederatedRecModel:
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
        - from FedFUGP:
            - fair_rho
            - fair_group_feature
            - from GeneralModelFairness
                - fair_lambda: the trade-off coefficient
        '''
        parser = FederatedRecModel.parse_model_args(parser)
        parser = Fed_FUGP.parse_fairness_args(parser)
        return parser
    
#     @staticmethod
#     def get_reader():
#         return "FedRecReader"
    
    def log(self):
        super().log()
        self.fair_controller.log()
        
    def __init__(self, args, reader, device):
        super().__init__(args, reader, device)
        self.fair_controller = Fed_FUGP(args, reader)
        
        
    ##################################
    #        federated control       #
    ##################################
    
    '''
    General federated learning with a central communicator:
    
    model.actions_before_train() --> internal calls:
                                        model.keep_cloud_params()
                                        model.fairness_controller.do_preprocess()
    for each epoch:
        model.actions_before_epoch() --> internal calls:
                                            model.fairness_controller.reset_statistics()
        for each user_batch:
            local_info = model.get_local_info(user_batch, {some training info})
            if not model.do_device_dropout():
                model.download_cloud_params()
                model.local_optimize() --> internal calls: 
                                                model.fairness_controller.do_in_epoch()
                                                model.upload_fairness_statistics()
                model.upload_edge_params()
        model.actions_after_epoch() --> internal calls:
                                            model.mitigate_params()
                                            model.fairness_controller.do_after_epoch()
    '''
    
    ##################################
    #        fairness control        #
    ##################################
    
#     def keep_cloud_params(self):
#         super().keep_cloud_params()
        
    def actions_before_train(self, info):
        # keep cloud parameters
        super().actions_before_train(info) 
        # fair preprocess
        if self.fair_controller.get_fairness_opt_type() == "preprocess":
            self.fair_controller.do_preprocess(self)
    
    def actions_before_epoch(self, info):
        # center node: reset group statistics at the beginning of each epoch
        # This will: 1) calculate prev_stats; 2) re-initialize statistics as zeros
        self.fair_controller.reset_statistics() 
        '''
        Also set up proposal for later aggregation
        '''
        
#     def get_local_info(self, feed_dict, training_info):
#         return super().get_local_info(feed_dict, training_info)
#     def download_cloud_params(self, local_info):
#         super().download_cloud_params(local_info)
    
    def local_optimize(self, feed_dict, local_info):
        '''
        Model optimization in local space
        '''
        learning_rate = local_info['lr']

        # get number of local rounds
        local_rounds = self.n_local_step if not self.random_local_step else np.random.randint(1,max(self.n_local_step+1,2))
    
        # local update
        local_loss = []
        for i in range(local_rounds):
            # local optimization
            self.optimizer.zero_grad()
            out_dict = self.forward(feed_dict)
            loss = self.get_loss(feed_dict, out_dict)
            
            # add fairness loss in training
            if self.fair_controller.get_fairness_opt_type() == "inepoch":
                fair_out = self.fair_controller.do_in_epoch(self, {'person': local_info['device'], 'local_step': i,
                                                                   'output': out_dict, 'loss': loss})
                loss = loss + fair_out['fair_loss'] 
                
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
            
        local_info['local_loss'] = local_loss[-1]
        if self.loss_type == "regression":
            raise NotImplemented
        else:
            return {"loss": np.mean(local_loss)}
    
    def upload_edge_params(self, local_info):
        # upload fairness statistics to central node
        if self.fair_controller.get_fairness_opt_type() == "inepoch":
            self.fair_controller.upload_fairness_statistics({'person': local_info['device'], 
                                                             'model': self, 'local_loss': local_info['local_loss']})
        '''
        Also upload edge parameters to cloud
        '''
        
#     def mitigate_params(self):
#         with torch.no_grad():
#             for name, param in self.cloud_params.items():
#                 if name in self.param_proposal:
#                     agg = self.cloud_params[name] * (1-self.mitigation_alpha) + \
#                             self.param_proposal[name] * self.mitigation_alpha / self.param_proposal_count[name]#.view(-1,1)
#                     select = self.param_proposal_count[name].view(-1) > 0
#                     self.cloud_params[name][select] = agg[select]