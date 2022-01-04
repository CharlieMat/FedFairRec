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

def init_ranking_report(k_list):
    report = {}
    for k in k_list:
        report["HR@%d"%k] = 0.
        report["P@%d"%k] = 0.
        report["RECALL@%d"%k] = 0.
        report["F1@%d"%k] = 0.
        report["NDCG@%d"%k] = 0.
    report["MR"] = 0.
    report["MRR"] = 0.
    report["AUC"] = 0.
    return report    

    
def calculate_ranking_metric(pos_pred, neg_pred, k_list, report = {}):
    '''
    @input:
    - pos_pred: (R,)
    - neg_pred: (L,)
    - k_list: e.g. [1,5,10,20,50]
    - report: {"HR@1": 0, "P@1": 0, ...}
    '''
    if len(report) == 0:
        report = init_ranking_report(k_list)
    R = pos_pred.shape[0] # number of positive samples
    L = neg_pred.shape[0] # number of negative samples
#     print(pos_pred)
#     print(neg_pred)
    N = R + L
    max_k = max(k_list)
    all_preds = torch.cat((pos_pred,neg_pred)).detach()
    topk_score, topk_indices = torch.topk(all_preds, N)
    ranks = torch.zeros(R)
    for i,idx in enumerate(topk_indices):
        if idx < R:
            ranks[idx] = i+1.
#     ranks = torch.sum(all_preds.view(1,-1) > pos_pred.view(R,1), dim = 1) + 1.
#     if all_preds.is_cuda:
#         all_preds = all_preds.detach().cpu()
#         ranks = ranks.detach().cpu()
#     print(ranks)
    # normalized mean rank
    mr = (torch.mean(ranks)/R).numpy()
    report["MR"] += mr
    # mean reciprocal rank
    report["MRR"] += torch.mean(1.0/ranks).numpy()
    # auc
    y = np.concatenate((np.ones(R),np.zeros(L)))
    auc = metrics.roc_auc_score(y, all_preds.cpu())
    report['AUC'] += auc
    # hit map of each position
    hitMap = torch.zeros(max_k)
    for i,idx in enumerate(torch.round(ranks).to(torch.long)):
        if idx <= max_k:
            hitMap[idx-1] = 1
#     print(hitMap)
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
#     print(f"HR:{hr[0]},P:{precision[0]},R:{recall[0]},NDCG:{ndcg[0]},MR:{mr},AUC:{auc}")
#     input()
    for k in k_list:
        report["HR@%d"%k] += hr[k-1]
        report["P@%d"%k] += precision[k-1]
        report["RECALL@%d"%k] += recall[k-1]
        report["F1@%d"%k] += f1[k-1]
        report["NDCG@%d"%k] += ndcg[k-1]
    return report
    
def userwise_ranking_eval_process(model, k_list, eval_sample_p, proc_id, n_proc):
    eval_loader = DataLoader(model.reader, batch_size = 1, shuffle = False)
    pbar = tqdm(total = model.reader.n_users)
    report = init_ranking_report(k_list)
    n_user_tested = 0
    with torch.no_grad():
        for i, batch_data in enumerate(eval_loader):
            if i % n_proc == proc_id:
                # sample user with record in eval data
                if np.random.random() <= eval_sample_p and "no_item" not in batch_data:
                    # predict
                    pos_probs, neg_probs = get_predictions_for_ranking_eval(model, batch_data)
                    # metrics
                    calculate_userwise_metric(pos_probs.view(-1), neg_probs.view(-1), k_list, report)
                    n_user_tested += 1
                pbar.update(1)
    pbar.close()
    for k,v in report.items():
        report[k] = v / (n_user_tested + 1e-7)
    pickle.dump(report, open(model.model_path + ".eval_" + str(proc_id), 'wb'))

class TopK(GeneralTask):
    
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
        self.eval_batch_size = 1  # userwise evaluation
        
    def log(self):
        super().log()
        print(f"\tat_k: {self.at_k_list}")
        print(f"\tn_eval_process: {self.n_eval_process}")
        print(f"\teval_batch_size: {self.eval_batch_size}")

    def do_epoch(self, model, epoch_id):
        model.reader.set_phase("train")
        train_loader = DataLoader(model.reader.get_train_dataset(), batch_size = self.batch_size, 
                                  shuffle = True, pin_memory = self.pin_memory,
                                  num_workers = model.reader.n_worker)
        torch.cuda.empty_cache()

        model.train()
        pbar = tqdm(total = len(train_loader.dataset))
        step_loss = []
        for i, batch_data in enumerate(train_loader):
            gc.collect()
            wrapped_batch = model.wrap_batch(batch_data)
            if i == 0 and epoch_id == 1:
                self.show_batch(wrapped_batch)
            result = model.do_forward_and_loss(wrapped_batch)
            step_loss.append(result["loss"].item())
            result["loss"].backward()
            model.optimizer.step()
            pbar.update(self.batch_size)
        pbar.close()
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
            report = self.evaluate_regression(model)
        else: # ranking evaluation
            report = self.evaluate_userwise_ranking(model)
        print("Result dict:")
        print(str(report))
        return report
    
    def evaluate_regression(self, model):
        '''
        Calculate rating metrics (RMSE, MAE)
        @input:
        - model: GeneralRecModel or its extension
            
        @output:
        - resultDict: {
            "rmse": rooted mean square error,
            "mae": mean absolute error,
            "metric": rmse
        }
        '''
        
        eval_loader = DataLoader(model.reader.get_eval_dataset(), worker_init_fn = worker_init_func,
                                 batch_size = 1, shuffle = False, pin_memory = False, num_workers = model.reader.n_worker)
        N = len(eval_loader.dataset)
        pbar = tqdm(total = N)
        report = {"MAE": 0., "RMSE": 0.}
        with torch.no_grad():
            for i, batch_data in enumerate(eval_loader):
                
                # predict
                wrapped_data = model.wrap_batch(batch_data)
                out_dict = model.forward(wrapped_data)
                preds = out_dict["preds"].view(-1)
                targets = feed_dict["resp"].view(-1)
                if preds.is_cuda:
                    preds = preds.detach()
                    targets = targets.detach()
                    
                # metrics
                diff = preds - targets
                ae = torch.mean(torch.abs(diff))
                se = torch.mean(diff ** 2)
                if preds.is_cuda:
                    ae = ae.cpu()
                    se = se.cpu()
                report["MAE"] += ae
                report["RMSE"] += se
                pbar.update(self.eval_batch_size)
        pbar.close()
        
        report["MAE"] = (report["MAE"]/N).numpy()
        report["RMSE"] = torch.sqrt(report["RMSE"]/N).numpy()
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
        with torch.no_grad():
            for i, batch_data in enumerate(eval_loader):
                # sample user with record in eval data
                if np.random.random() <= self.eval_sample_p and "no_item" not in batch_data:
                    # predict
                    pos_probs, neg_probs = self.get_predictions_for_ranking_eval(model, batch_data)
                    # metrics
                    calculate_ranking_metric(pos_probs.view(-1), neg_probs.view(-1), self.at_k_list, report)
                    n_user_tested += 1
        for key, value in report.items():
            report[key] /= n_user_tested
        return report
    
    def get_predictions_for_ranking_eval(self, model, batch_data):
        feed_dict = model.wrap_batch(batch_data)
        out_dict = model.forward(feed_dict, return_prob = True)
        pos_preds, neg_preds = out_dict["probs"], out_dict["neg_probs"]
        if pos_preds.is_cuda:
            pos_preds = pos_preds.detach().cpu()
            neg_preds = neg_preds.detach().cpu()
        return pos_preds, neg_preds
    

#     def evaluate_userwise_ranking_multi_process(self, model):
#         offset = len(model.reader) // self.n_eval_process
#         # map to processes
#         processList = [Process(target=userwise_ranking_eval_process, 
#                            args=(model, self.at_k_list, self.eval_sample_p, i, self.n_eval_process)) \
#                    for i in range(self.n_eval_process)]
#         for i in range(self.n_eval_process):
#             processList[i].start()
#         for i in range(self.n_eval_process):
#             processList[i].join()
#         # reduce
#         reports = [pickle.load(open(model.model_path + ".eval_" + i, 'rb') for i in range(self.n_eval_process))]
#         combined_report = {k: np.mean([report[k] for report in reports]) for k in report[0].keys()}
#         return combiend_report
    