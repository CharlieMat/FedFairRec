import os
import gc
import heapq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import pickle
from multiprocessing import Pool, Process
# import threading
# from concurrent.futures import ThreadPoolExecutor
# from tqdm_multi_thread import TqdmMultiThreadFactory

import utils
from task.GeneralTask import GeneralTask
from reader.BaseReader import worker_init_func
from task.TopK import init_ranking_report, calculate_ranking_metric

def update_fairness_group_result(final_report_dict, group_name, v, new_report):
    '''
    Update the evaluated fairness in final_report_dict by new_report.
    Each corresponding metric is maintained as (val, count)
    @input:
    - final_report_dict: {group_key: {group_value: {metric: statistics}}}, here statiistics = (sum, count)
    - group_name: the group feature to update
    - v: the corresponding value under the group feature
    - new_report: {metric: evaluated value}
    '''
    if v not in final_report_dict[group_name]:
        final_report_dict[group_name][v] = {metric:(val,1) for metric,val in new_report.items()}
    else:
        for metric,val in new_report.items():
             final_report_dict[group_name][v][metric] = (final_report_dict[group_name][v][metric][0] + val, 
                                                         final_report_dict[group_name][v][metric][1] + 1)  
    
def get_activity_info(rec_reader, phase = "train", feature = "UserID"):
    '''
    Get activity information of user/item
    @output:
    - active_group_ids: [user/item id]
    - inactive_group_ids: [user/item id], is mutually exclusive with active_group_ids
    - act_dict: {user/item id: count/frequency}
    - threshold: scalar, the average count
    '''
#     act_dict = {}
    act_dict = rec_reader.data[phase][feature].value_counts().to_dict()
    if feature == "UserID":
        for uid in rec_reader.users:
            if uid not in act_dict:
                act_dict[uid] = 0
        act_dict = {rec_reader.get_user_feature(uid, "UserID"): count for uid, count in act_dict.items()}
    elif feature == "ItemID":
        for iid in rec_reader.items:
            if iid not in act_dict:
                act_dict[iid] = 0
        act_dict = {rec_reader.get_item_feature(uid, "ItemID"): count for iid, count in act_dict.items()}
    else:
        raise NotImplemented
    avg_count = sum(act_dict.values()) / len(act_dict) # average count as threshold
    active_group = [k for k,v in D.items() if v > avg_count]
    inactive_group = [k for k,v in D.items() if v <= avg_count]
    return [idx for _,idx in active_group], inactive_group, act_dict, avg_count

class UserGroupFairnessEval(GeneralTask):
    
    @staticmethod
    def parse_task_args(parser):
        '''
        - at_k
        - n_eval_process
        - user_group_field
        - args from GeneralTask:
            - optimizer
            - epoch
            - check_epoch
            - lr
            - batch_size
            - eval_batch_size
            - with_val
            - with_test
            - val_sample_p
            - test_sample_p
            - stop_metric
            - pin_memory
        '''
        parser = GeneralTask.parse_task_args(parser)
        parser.add_argument('--at_k', type=int, nargs='+', default=[1,5,10,20,50], 
                            help='specificy a list of k for top-k performance')
        parser.add_argument('--n_eval_process', type=int, default=1, 
                            help='number of thread during eval')
        parser.add_argument('--user_group_field', type=str, nargs='+', default=['activity'])
#         parser.add_argument('--item_group_field', type=str, nargs='+', default=[])
        return parser
    
    def __init__(self, args, reader):
        self.at_k_list = args.at_k
        self.n_eval_process = args.n_eval_process
        self.user_group_field = args.user_group_field
        self.group_for_evaluation = self.user_group_field[0]
#         self.item_group_field = args.item_group_field
        super().__init__(args, reader)
        self.eval_batch_size = 1  # userwise evaluation
        
    def log(self):
        super().log()
        print(f"\tat_k: {self.at_k_list}")
        print(f"\tn_eval_process: {self.n_eval_process}")
        print(f"\tuser_group_field: {self.user_group_field}")
        print(f"\teval_batch_size: {self.eval_batch_size}")
        
    def train(self, model, continuous = False):
        pass

    def do_eval(self, model):
        """
        Evaluate the results for an eval dataset.
        @input:
        - model: GeneralRecModel or its extension
        
        @output:
        - resultDict: {metric_name: metric_value}
        """

        print("Evaluating...")
        print("Sample p = " + str(self.eval_sample_p))
        model.eval()
        if model.loss_type == "regression": # rating prediction evaluation
#             report = self.evaluate_regression(model)
            raise NotImplemented
        else: # ranking evaluation
            report = self.evaluate_fairness(model)
        print("Result dict:")
        print(str(report))
        return report

    def add_fairness_evaluation(self, model):
        '''
        Calculate ranking metrics

        @input:
        - model: GeneralRecModel or its extension
        
        @output:
        - resultDict:{"user_group_xxx": {feature_value: {metric: average evaluated value}}}
        '''
        eval_data = model.reader.get_eval_dataset()
        eval_loader = DataLoader(eval_data, worker_init_fn = worker_init_func,
                                 batch_size = 1, shuffle = False, pin_memory = False, 
                                 num_workers = eval_data.n_worker)
        
        report = {}
        for uG in self.user_group_field:
            report[f"user_group_{uG}"] = {}
        print(f"Fairness observation:{report.keys()}")
        
        uA,uI,u_activity_dict,uT = get_activity_info(model.reader, side = "user")
#         iA,iI,i_activity_dict,iT = get_activity_info(model.reader, side = "item")
        
        with torch.no_grad():
            for i, batch_data in enumerate(eval_loader):
                if np.random.random() <= self.eval_sample_p and "no_item" not in batch_data:
                    uid = batch_data["user_UserID"].reshape(()).detach().cpu().numpy()
                    # predict
                    feed_dict = model.wrap_batch(batch_data)
                    out_dict = model.forward(feed_dict, return_prob = True)
                    pos_probs, neg_probs = out_dict["probs"], out_dict["neg_probs"]
                    # metrics
                    ranking_report = calculate_ranking_metric(pos_probs.view(-1), neg_probs.view(-1), self.at_k_list)
                    user_meta = model.reader.get_user_meta(model.reader.users[uid])
                    # record evaluation info for user groups of each selected field
                    for user_field in self.user_group_field:
                        if user_field in user_meta:
                            v = user_meta[user_field]
                        elif user_field == "activity":
                            v = "inactive" if u_activity_dict[uid] < uT else "active"
                        else:
                            raise NotImplemented
                        update_fairness_group_result(report, f"user_group_{user_field}", v, ranking_report)
        # aggregate each metric
        # {"user_group_xxx": {feature_value: {metric: (value_sum, value_count)}}}
        # --> {"user_group_xxx": {feature_value: {metric: average evaluated value}}}
        for group_name, group_dict in report.items():
            for group_id, metric_dict in group_dict.items():
                group_dict[group_id] = {metric: val_tuple[0]/val_tuple[1] if val_tuple[1] !=0 else 0 for metric, val_tuple in metric_dict.items()}
        return report


    

    