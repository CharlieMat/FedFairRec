import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from reader.BaseReader import worker_init_func
from task.TopK import calculate_ranking_metric
from model.fair_rec.GeneralModelFairness import GeneralModelFairness

def get_user_activity_info(rec_reader, phase = "train"):
    '''
    Get activity information of user/item
    @output:
    - active_group_ids: [user id]
    - inactive_group_ids: [user id], is mutually exclusive with active_group_ids
    - act_dict: {user id: count/frequency}
    - threshold: scalar, the average count
    '''
    act_dict = rec_reader.data[phase]["UserID"].value_counts().to_dict()
    for uid in rec_reader.users:
        if uid not in act_dict:
            act_dict[uid] = 0
    # raw UserID --> encoded uid
    act_dict = {rec_reader.get_user_feature(uid, "UserID"): count for uid, count in act_dict.items()}
    # average count as threshold
    avg_count = sum(act_dict.values()) / len(act_dict) 
    # separate active and inactive group
    active_group = [uid for uid,count in act_dict.items() if count > avg_count]
    inactive_group = [uid for uid,count in act_dict.items() if count <= avg_count]
    return active_group, inactive_group, act_dict, avg_count

def get_user_group_info(rec_reader, group_feature):
    if group_feature == 'activity':
        uA,uI,u_activity_dict,uT = get_user_activity_info(rec_reader)
        print(f"user activity: {len(uA)}(A) -- {len(uI)}(I), threshold ({uT})")
        group_dict = {uid: 'active' if freq > uT else 'inactive' for uid,freq in u_activity_dict.items()}
        feature_values = ['active', 'inactive']
    else:
        df = pd.read_table(rec_reader.user_meta_file, sep = '\t', engine = 'python')
        id_name = df.columns[0]
        df.index = df[id_name]
        meta = df.to_dict(orient = 'index')
        group_dict,group_count = {},{}
        for uid, meta_features in meta.items():
            user_group = meta_features[group_feature]
            group_dict[rec_reader.get_user_feature(uid, "UserID")] = user_group
            if user_group not in group_count:
                group_count[user_group] = 1
            else:
                group_count[user_group] += 1
        print(f"user {group_feature}: {group_count}")
        feature_values = list(set(group_dict.values()))
    return group_dict, feature_values
    

def get_user_eval(pos_pred, neg_pred, k_list = [1,5,10,20,50]):
    '''
    @input:
    - pos_pred: (R,)
    - neg_pred: (L,)
    - k_list: e.g. [1,5,10,20,50]
    - report: {"HR@1": 0, "P@1": 0, ...}
    '''
    report = {}
    pos_pred = pos_pred.view(-1)
    neg_pred = neg_pred.view(-1)
    R = pos_pred.shape[0] # number of positive samples
    L = neg_pred.shape[0] # number of negative samples
    N = R + L
    max_k = max(k_list)
    all_preds = torch.cat((pos_pred,neg_pred)).detach()
    topk_score, topk_indices = torch.topk(all_preds, N)
    ranks = torch.zeros(R)
    for i,idx in enumerate(topk_indices):
        if idx < R:
            ranks[idx] = i+1.
    # hit map of each position
    hitMap = torch.zeros(max_k)
    for i,idx in enumerate(torch.round(ranks).to(torch.long)):
        if idx <= max_k:
            hitMap[idx-1] = 1
    # hit ratio, recall, f1, ndcg
    tp = torch.zeros(max_k) # true positive
    tp[0] = hitMap[0]
    dcg = torch.zeros(N) # DCG
    dcg[0] = hitMap[0]
    idcg = torch.zeros(N) # IDCG
    idcg[0] = 1
    for i in range(1,max_k):
        tp[i] = tp[i-1] + hitMap[i]
        b = torch.tensor(i+2).to(torch.float) # pos + 1 = i + 2
        dcg[i] = dcg[i-1] + hitMap[i]/torch.log2(b)
        idcg[i] = idcg[i-1] + 1.0/torch.log2(b) if i < R else idcg[i-1]
    hr = tp.clone().numpy()
    hr[hr>0] = 1
    precision = (tp / torch.arange(1, max_k+1).to(torch.float)).numpy()
    recall = (tp / R).numpy()
    f1 = (2*tp / (torch.arange(1, max_k+1).to(torch.float) + R)).numpy() # 2TP / ((TP+FP) + (TP+FN))
    ndcg = (dcg / idcg).numpy()
    for k in k_list:
        report["HR@%d"%k] = hr[k-1]
        report["P@%d"%k] = precision[k-1]
        report["RECALL@%d"%k] = recall[k-1]
        report["F1@%d"%k] = f1[k-1]
        report["NDCG@%d"%k] = ndcg[k-1]
    return report
    

class FairUserGroupPerformance(GeneralModelFairness):
    '''
    Controller for performance-based user group fairness
    - metric = mean_{G,G'}(|mean_{u in G}(performance of u) - mean_{u in G'}(performance of u)|^rho)
    - Note: performance can be substitute by (1 - loss)

    user_wise gradient/loss scalar caused by the fairness metric:
    loss *= (1. - lambda * rho * (1 if u in superior group else -1) * |A - B)|^{rho-1})
    where A = mean_{u in G}(performance of u) and B = mean_{u not in G}(performance of u) are stored in previous epoch

    Fairness aware task can include this controller as additional optimization tool by setting the following:
    - Include controller.parse_model_args(parser) in the task's parse_model_args() function
    - Initialize the controller in task's initialization step
    - Call controller.reset_statistics at the beginning of each epoch
    - In each batch training, add controller.get_loss() on the model loss before optimizer.step()
    - In task's do_eval, call controller.add_fairness_evaluation() after evaluating recommendation performance
    '''
    
    @staticmethod
    def parse_fairness_args(parser):
        '''
        args:
        - fair_rho
        - fair_group_feature
        - from GeneralModelFairness
            - fair_lambda: the trade-off coefficient
        '''
        parser = GeneralModelFairness.parse_fairness_args(parser)
        parser.add_argument('--fair_rho', type=int, default=1, 
                            help='1: absolute group difference; 2: squared group difference')
        parser.add_argument('--fair_group_feature', type=str, default='activity', 
                            help='e.g. "Gender", "activity"')
        return parser
    
    def get_fairness_opt_type(self):
        return "inepoch" # one of {preprocess, inepoch, afterepoch, postprocess}
    
    def log(self):
        super().log()
        print(f"\tfair_rho: {self.fair_rho}")
        print(f"\tfair_group_feature: {self.group_feature}")
    
    def __init__(self, args, reader):
        GeneralModelFairness.__init__(self, args, reader)
        self.fair_rho = args.fair_rho
        self.group_feature = args.fair_group_feature
        # group feature of all users
        GD, F = get_user_group_info(reader, self.group_feature)
        self.group_dict = GD
        self.feature_values = F
        assert self.feature_values and len(self.feature_values) >= 2
        # set up sufficient statistics
        self.prev_statistics = {v: np.random.random() for v in self.feature_values}
        self.statistics = {"sum": {v: 0.5 for v in self.feature_values}, 
                           "count": {v: 1 for v in self.feature_values}}
        
        
    def reset_statistics(self):
        # sufficient statistics of feature_values: sum and count of each group value
        print(self.statistics)
        self.prev_statistics = {v: self.statistics['sum'][v] / self.statistics['count'][v] for v in self.feature_values}
        print(f"Previous statistics:\n{self.prev_statistics}")
        self.statistics = {"sum": {v: 0. for v in self.feature_values}, 
                           "count": {v: 0 for v in self.feature_values}}
        
    def get_eval(self):
        S = []
        for i,v0 in enumerate(self.feature_values):
            for v1 in self.feature_values[i+1:]:
                S.append(abs(self.statistics['sum'][v0]/self.statistics['count'][v0] - 
                             self.statistics['sum'][v1]/self.statistics['count'][v1]) ** self.fair_rho)
        return {f"{self.group_feature}": np.mean(S)}
    
    def do_in_epoch(self, model, batch_info):
        '''
        fairness_loss = loss * (-lambda * rho * (1 if u in superior group else -1) * |A - B)|^{rho-1})
        * A = mean_{u in G}(performance of u)
        * B = mean_{u not in G}(performance of u)
        '''
        feed_dict = batch_info['batch']
        out_dict = batch_info['output']
        loss = batch_info['loss']
        selected_metric = batch_info['metric']
        
        uid = feed_dict["user_UserID"].reshape(-1).detach().cpu().numpy()[0]
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
        fair_loss = - loss * (group_difference / len(self.feature_values))
        self.statistics['sum'][G] += 1.-loss.item()
        self.statistics['count'][G] += 1
        return {'fair_loss': fair_loss}
    
    
    def add_fairness_evaluation(self, model, params, method = 'diff'):
        '''
        Calculate ranking metrics

        @input:
        - model: GeneralRecModel or its extension
        - params: {selected_metric, at_k_list, eval_sample_p}
        
        @output:
        - resultDict:{"user_group_xxx": {feature_value: {metric: average evaluated value}}}
        '''
        selected_metric = params["selected_metric"] if 'selected_metric' in params else "AUC" 
        at_k_list = params['at_k_list'] if 'at_k_list' in params else [1,10,50]
        eval_sample_p = params['eval_sample_p'] if 'eval_sample_p' in params else 1.0
        
        print("Fairness evaluation:")
        eval_data = model.reader.get_eval_dataset()
        eval_loader = DataLoader(eval_data, worker_init_fn = worker_init_func,
                                 batch_size = 1, shuffle = False, pin_memory = False, 
                                 num_workers = eval_data.n_worker)
        
#         report = {group_name: {v: {} for v in self.feature_values}}
#         print(f"\tGroup feature:{report.keys()}")
        with torch.no_grad():
            for i, batch_data in enumerate(eval_loader):
                if np.random.random() <= eval_sample_p and "no_item" not in batch_data:
                    uid = batch_data["user_UserID"].reshape(-1).detach().cpu().numpy()[0]
                    G = self.group_dict[uid]
                    # predict
                    feed_dict = model.wrap_batch(batch_data)
                    out_dict = model.forward(feed_dict, return_prob = True)
#                     loss = model.get_loss(wrapped_batch, out_dict)
#                     pos_probs, neg_probs = out_dict["probs"], out_dict["neg_probs"]
                    # metrics
#                     user_report = get_user_eval(pos_probs.view(-1), neg_probs.view(-1), at_k_list)
#                     ranking_report = calculate_ranking_metric(pos_probs.view(-1), neg_probs.view(-1), at_k_list)
                    # record evaluation info for user groups of each selected field
#                     for metric,val in ranking_report.items():
#                         if metric not in report[group_name][G]:
#                             report[group_name][G][metric] = (val,1)
#                         else:
#                             report[group_name][G][metric] = (report[group_name][G][metric][0] + val, 
#                                                              report[group_name][G][metric][1] + 1) 
#                     self.statistics['sum'][G] += (1. - loss.item())
#                     self.statistics['sum'][G] += user_report[selected_metric]
#                     self.statistics['count'][G] += 1
        # aggregate each metric
        # {"user_group_xxx": {feature_value: {metric: (value_sum, value_count)}}}
        # --> {"user_group_xxx": {feature_value: {metric: value_sum / value_count}}}
        S = []
        for i,v0 in enumerate(self.feature_values):
            for v1 in self.feature_values[i+1:]:
                S.append(abs(self.statistics['sum'][v0]/self.statistics['count'][v0] - 
                             self.statistics['sum'][v1]/self.statistics['count'][v1]) ** self.fair_rho)
#         group_dict = report[group_name]
#         for group_value, metric_dict in group_dict.items():
#             group_dict[group_value] = {metric: val_tuple[0]/val_tuple[1] if val_tuple[1] !=0 else 0 \
#                                        for metric, val_tuple in metric_dict.items()}
#         group_name = f"user_group_{self.group_feature}"
        return {f"{self.group_feature}_{selected_metric}": np.mean(S)}
        