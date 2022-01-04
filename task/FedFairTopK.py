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

from reader.BaseReader import worker_init_func
from task.TopK import init_ranking_report, calculate_ranking_metric
from task.FedTopK import FedTopK
from model.fair_rec.Fed_FUGP import Fed_FUGP

class FedFairTopK(FedTopK):
    
    def evaluate_userwise_ranking(self, model):
        report = super().evaluate_userwise_ranking(model)
        params = {'selected_metric': self.stop_metric, 'at_k_list': self.at_k_list, 'eval_sample_p': self.eval_sample_p}
        fairness_report = model.fair_controller.add_fairness_evaluation(model, params)
        for k,v in fairness_report.items():
            report["fair_" + k] = v
        report[self.stop_metric] += self.stop_metric_sign * fairness_report[self.stop_metric]
        return report


    
    
        
    