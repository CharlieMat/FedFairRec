import os
import gc
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


class FedTopK(GeneralTask):
    
    @staticmethod
    def parse_task_args(parser):
        '''
        - at_k
        - n_eval_process
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
        return parser
    
    def __init__(self, args, reader):
        self.at_k_list = args.at_k
        self.n_eval_process = args.n_eval_process
        super().__init__(args, reader)
        self.batch_size = 1 # userwise training
        self.eval_batch_size = 1  # userwise evaluation
        
    def log(self):
        super().log()
        print(f"\tat_k: {self.at_k_list}")
        print(f"\tn_eval_process: {self.n_eval_process}")
        print(f"\teval_batch_size: {self.eval_batch_size}")

    def do_epoch(self, model, epoch_id):
        model.reader.set_phase("train")
        train_data = model.reader.get_train_dataset()
        train_loader = DataLoader(train_data, worker_init_fn = worker_init_func,
                                  batch_size = 1, shuffle = False, pin_memory = False, 
                                  num_workers = train_data.n_worker)
        torch.cuda.empty_cache()

        model.train()
        step_loss = []
        dropout_count = 0
        for i, batch_data in enumerate(train_loader):
#             if i > 3000:
#                 break
            gc.collect()
            if "no_item" in batch_data:
                continue
            wrapped_batch = model.wrap_batch(batch_data)
            if i == 0 and epoch_id == 1:
                self.show_batch(wrapped_batch)
            local_info = model.get_local_info(wrapped_batch, {'epoch':epoch_id, 'lr': self.lr})
            
            # imitate user dropout in FL (e.g. connection lost or no response)
            if model.do_device_dropout(local_info):
                dropout_count += 1
                continue
            
            # download domain-specific mapping models to personal spaces
            model.download_cloud_params(local_info)
            
            # local optimization
            local_response = model.local_optimize(wrapped_batch, local_info) 
            step_loss.append(local_response["loss"])
            
            # upload updated domain-specific mapping models to the cloud of each domain
            model.upload_edge_params(local_info)
            
        model.download_cloud_params(None) # synchronize parameter for model saving
        print(f"#dropout device: {dropout_count}")
        return {"loss": np.mean(step_loss), "step_loss": step_loss}

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
            report = self.evaluate_userwise_ranking(model)
        print("Result dict:")
        print(str(report))
        return report

    def evaluate_userwise_ranking(self, model):
        '''
        Calculate ranking metrics

        @input:
        - model: GeneralRecModel or its extension
        
        @output:
        - resultDict:
        {
            "mr": mean rank
            "mrr": mean reciprocal rank
            "auc": area under the curve
            "hr": [1 if a hit @k] of size (L+1), take the mean over all records
            "p": precision
            "recall": recall
            "f1": f1-score
            "ndcg": [DCG score] since IDCG=1 for each record. DCG = 1/log(rank) after rank, =0 before rank
            "metric": auc
        }
        '''
        eval_data = model.reader.get_eval_dataset()
        eval_loader = DataLoader(eval_data, worker_init_fn = worker_init_func,
                                 batch_size = 1, shuffle = False, pin_memory = False, 
                                 num_workers = eval_data.n_worker)
        report = init_ranking_report(self.at_k_list)
        n_user_tested = 0
#         model.keep_cloud_params() # store parameters on central server
        with torch.no_grad():
            for i, batch_data in enumerate(eval_loader):
                # sample user with record in eval data
                if np.random.random() <= self.eval_sample_p and "no_item" not in batch_data:
                    feed_dict = model.wrap_batch(batch_data)
                    # download domain-specific mapping models to personal spaces
                    local_info = model.get_local_info(feed_dict, {'epoch':0, 'lr': 0.})
                    model.download_cloud_params(local_info) 
                    # predict
                    out_dict = model.forward(feed_dict, return_prob = True)
                    pos_probs, neg_probs = out_dict["probs"].view(-1), out_dict["neg_probs"].view(-1)
                    # metrics
                    calculate_ranking_metric(pos_probs, neg_probs, self.at_k_list, report)
                    n_user_tested += 1
        for key, value in report.items():
            report[key] /= n_user_tested
        return report
    
    
        
    