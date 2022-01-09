import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from reader.BaseReader import worker_init_func
from task.TopK import calculate_ranking_metric
from model.fair_rec.FairUserGroupPerformance import FairUserGroupPerformance


class Fed_FUGP(FairUserGroupPerformance):
    '''
    Controller for federated FUGP(FairUserGroupPerformance)

    Fairness aware task can include this controller as additional optimization tool by setting the following:
    - Include controller.parse_model_args(parser) in the task's parse_model_args() function
    - Initialize the controller in task's initialization step
    - Call controller.reset_statistics at the beginning of each epoch
    - In each user's local training, add controller.add_local_regularization() on the model before uploading model parameters
    - In each user's local training, add controller.upload_fairness_statistics() along with model upload
    - In task's do_eval, call controller.add_fairness_evaluation() after evaluating recommendation performance
    '''
    
    @staticmethod
    def parse_fairness_args(parser):
        '''
        args:
        - fair_noise_sigma
        - from FairUserGroupPerformance:
            - fair_rho
            - fair_group_feature
            - from GeneralModelFairness:
                - fair_lambda
        '''
        parser = FairUserGroupPerformance.parse_fairness_args(parser)
        parser.add_argument('--fair_noise_sigma', type=float, default=1.0, 
                            help='The variance for noise signals of fairness statistics')
        return parser
    
    def __init__(self, args, reader):
        '''
        - fair_lambda, fair_rho, group_feature
        - group_dict: {uid: feature_value}
        - feature_values: [feature_value]
        - prev_statistics: {feature_value: scalar}
        - statistics: {'sum': {feature_value: scalar}, 'count': {feature_value: scalar}}
        - group_ids: {feature_value: feature_id}
        - personal_sum_noise: {uid: [sigma]}
        - personal_count_noise: {uid: [sigma]}
        '''
        super(Fed_FUGP,self).__init__(args, reader)
        self.fair_noise_sigma = args.fair_noise_sigma
        # set up userwise noise signal
        self.group_ids = {fv: i for i,fv in enumerate(self.feature_values)}
        self.personal_sum_noise = {uid: np.random.randn(len(self.feature_values)) * self.fair_noise_sigma \
                                    for uid in range(reader.n_users)} # {uid: [epsilon(G0,uid),epsilon(G1,uid),...]}
        self.personal_count_noise = {uid: np.random.randn(len(self.feature_values)) * self.fair_noise_sigma \
                                    for uid in range(reader.n_users)} # {uid: [epsilon(G0,uid),epsilon(G1,uid),...]}
        
    def log(self):
        super().log()
        print(f"\tfair_noise_sigma: {self.fair_noise_sigma}")
    
#     def do_in_epoch(self, model, batch_info):
#         '''
#         fairness_loss = loss * (-lambda * rho * (1 if u in superior group else -1) * |A - B|^{rho-1})
#         * A = mean_{u in G}(performance of u)
#         * B = mean_{u in some group other than G}(performance of u)
#         '''
#         uid = batch_info['person']
#         loss = batch_info['loss']
        
#         if uid not in self.group_dict:
#             return 0
#         G = self.group_dict[uid] # the user's group
#         A = self.prev_statistics[G] # previous statistics of all groups
#         group_difference = 0.
#         for v,B in self.prev_statistics.items():
#             if v != G:
#                 C = self.fair_rho if A > B else -self.fair_rho
#                 scalar = self.fair_lambda * C * (abs(A-B) ** (self.fair_rho - 1))
#                 group_difference += scalar
#         fair_loss = - loss * (group_difference / len(self.feature_values))
#         return {'fair_loss': fair_loss}
    
    def do_in_epoch(self, model, local_info):
        '''
        @input:
        - local_info: {'device', 'user', 'item', 'negitem', 'epoch', 'lr'}
        '''
        uid = local_info['device']
        
        if uid not in self.group_dict:
            return 0
        G = self.group_dict[uid] # the user's group
        A = self.prev_statistics[G] # previous statistics of all groups
        group_difference = 0.
        for v,B in self.prev_statistics.items():
            if v != G:
                C = self.fair_rho if A > B else -self.fair_rho
                scalar = self.fair_lambda * C * (abs(A-B) ** (self.fair_rho - 1))
                group_difference += scalar
        D = - group_difference / len(self.feature_values) + 1
        # regulate gradient
        with torch.no_grad():
            for name, param in model.cloud_params.items():
                if name in model.param_proposal:
                    gradient = param.data - model.cloud_params[name]
                    param.data = model.cloud_params[name] + D * gradient
    
    def upload_fairness_statistics(self, local_info):
        uid = local_info['device']
        G_u = self.group_dict[uid]
        F = 1. - local_info['loss']
        # upload statistics
        for i,G in enumerate(self.feature_values):
            if G_u == G:
                self.statistics['sum'][G] += F \
                                                + self.personal_sum_noise[uid][i] \
                                                + np.random.randn() * self.fair_noise_sigma
                self.statistics['count'][G] += 1 + self.personal_count_noise[uid][i]
            else:
                self.statistics['sum'][G] += 0. \
                                                + self.personal_sum_noise[uid][i] \
                                                + np.random.randn() * self.fair_noise_sigma
                self.statistics['count'][G] += 0. + self.personal_count_noise[uid][i]
        