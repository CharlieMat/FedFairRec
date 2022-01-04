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
        self.statistics = {"sum": {v: 0. for v in self.feature_values}, 
                           "count": {v: 1 for v in self.feature_values}}
        
        
    def reset_statistics(self):
        # sufficient statistics of feature_values: sum and count of each group value
        print(self.statistics)
        self.prev_statistics = {v: self.statistics['sum'][v] / self.statistics['count'][v] for v in self.feature_values}
        self.statistics = {"sum": {v: 0. for v in self.feature_values}, 
                           "count": {v: 0 for v in self.feature_values}}
    
    def do_in_epoch(self, model, batch_info):
        '''
        fairness_loss = loss * (-lambda * rho * (1 if u in superior group else -1) * |A - B)|^{rho-1})
        * A = mean_{u in G}(performance of u)
        * B = mean_{u not in G}(performance of u)
        '''
        feed_dict = batch_info['batch']
        out_dict = batch_info['output']
        loss = batch_info['loss']
        
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
        self.statistics['sum'][G] += (1. - loss.item())
        self.statistics['count'][G] += 1
        fair_loss = - loss * (group_difference / len(self.feature_values))
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
        
        group_name = f"user_group_{self.group_feature}"
        report = {group_name: {v: {} for v in self.feature_values}}
        print(f"\tGroup feature:{report.keys()}")
        with torch.no_grad():
            for i, batch_data in enumerate(eval_loader):
                if np.random.random() <= eval_sample_p and "no_item" not in batch_data:
                    uid = batch_data["user_UserID"].reshape(-1).detach().cpu().numpy()[0]
                    # predict
                    feed_dict = model.wrap_batch(batch_data)
                    out_dict = model.forward(feed_dict, return_prob = True)
                    pos_probs, neg_probs = out_dict["probs"], out_dict["neg_probs"]
                    # metrics
                    ranking_report = calculate_ranking_metric(pos_probs.view(-1), neg_probs.view(-1), at_k_list)
                    # record evaluation info for user groups of each selected field
                    G = self.group_dict[uid]
                    for metric,val in ranking_report.items():
                        if metric not in report[group_name][G]:
                            report[group_name][G][metric] = (val,1)
                        else:
                            report[group_name][G][metric] = (report[group_name][G][metric][0] + val, 
                                                             report[group_name][G][metric][1] + 1)  
        # aggregate each metric
        # {"user_group_xxx": {feature_value: {metric: (value_sum, value_count)}}}
        # --> {"user_group_xxx": {feature_value: {metric: value_sum / value_count}}}
        group_dict = report[group_name]
        for group_value, metric_dict in group_dict.items():
            group_dict[group_value] = {metric: val_tuple[0]/val_tuple[1] if val_tuple[1] !=0 else 0 \
                                       for metric, val_tuple in metric_dict.items()}
        # calculate fairness evaluation metric
#         aggregate_report = {metric: [group_dict[G][metric] for G in self.feature_values] \
#                             for metric in group_dict[self.feature_values[0]]}
        for metric in group_dict[self.feature_values[0]]:
            performance_list = [group_dict[G][metric] for G in self.feature_values]
            F = []
            if method == 'diff':
                max_abs = max(abs(max(performance_list)), abs(min(performance_list))) + 1e-7
                for i,A in enumerate(performance_list):
                    for j in range(i+1,len(performance_list)):
                        F.append(abs(A-performance_list[j]) / max_abs)
            elif method == 'original':
                for i,A in enumerate(performance_list):
                    for j in range(i+1,len(performance_list)):
                        F.append(self.fair_lambda * (abs(A-performance_list[j]) ** self.fair_rho))
            report[metric] = np.mean(F)
        assert selected_metric in report
        return report
        