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
from task.TopK import TopK, init_ranking_report, calculate_ranking_metric
from reader.BaseReader import worker_init_func
from model.fair_rec.FairUserGroupPerformance import FairUserGroupPerformance
from reader.RecDataReader import RecUserReader

class FairTopK(TopK):
    
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
        parser = TopK.parse_task_args(parser)
        parser = FairUserGroupPerformance.parse_fairness_args(parser)
        return parser
    
    def log(self):
        super().log()
        self.fair_controller.log()
    
    def __init__(self, args, reader):
        super().__init__(args, reader)
        self.batch_size = 1 # userwise training
        self.fair_controller = FairUserGroupPerformance(args, reader)

    def do_epoch(self, model, epoch_id):
        model.reader.set_phase("train")
        train_reader = RecUserReader(model.reader, phase = "train", n_neg = model.reader.n_neg)
        train_loader = DataLoader(train_reader, batch_size = self.batch_size, 
                                  shuffle = True, pin_memory = self.pin_memory,
                                  num_workers = model.reader.n_worker)
        torch.cuda.empty_cache()

        model.train()
        pbar = tqdm(total = len(train_loader.dataset))
        step_loss = []
        for i, batch_data in enumerate(train_loader):
            gc.collect()
            if "no_item" in batch_data:
                pbar.update(self.batch_size)
                continue
            wrapped_batch = model.wrap_batch(batch_data)
            if i == 0 and epoch_id == 1:
                self.show_batch(wrapped_batch)
            out_dict = model.forward(wrapped_batch)
            loss = model.get_loss(wrapped_batch, out_dict)
            if self.fair_controller.get_fairness_opt_type() == "inepoch":
                fair_out = self.fair_controller.do_in_epoch(model, {'batch': wrapped_batch, 'output': out_dict, 
                                                                    'loss': loss, 'metric': self.stop_metric})
                loss = loss + fair_out['fair_loss'] # add fairness loss in training
            step_loss.append(loss.item())
            loss.backward()
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
#             if model.reader.phase == "test":
#             params = {'selected_metric': self.stop_metric, 'at_k_list': self.at_k_list, 'eval_sample_p': self.eval_sample_p}
#             fairness_report = self.fair_controller.add_fairness_evaluation(model, params)
#             for k,v in fairness_report.items():
#                 report["fair_" + k] = v
        print("Result dict:")
        print(str(report))
        return report
    
    def get_before_train_info(self, model):
        if self.fair_controller.get_fairness_opt_type() == "preprocess":
            self.fair_controller.do_preprocess(model)
            
    def get_after_epoch_info(self, model):
        if self.fair_controller.get_fairness_opt_type() == "afterepoch":
            self.fair_controller.do_after_epoch(model)
        self.fair_controller.reset_statistics() # reset group statistics at the beginning of each epoch