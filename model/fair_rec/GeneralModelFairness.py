import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from reader.BaseReader import worker_init_func
from task.TopK import calculate_ranking_metric
    

class GeneralModelFairness(object):
    '''
    Controller model fairness. Three types in general:
    - Preprocessing
    - Inprocess
    - Postprocessing

    Fairness aware task can include this controller as additional optimization tool by setting the following:
    - Include controller.parse_model_args(parser) in the task's parse_model_args() function
    - Initialize the controller in task's initialization step
    - Call controller.reset_statistics at the beginning of each epoch
    - In task's do_eval, call controller.add_fairness_evaluation() after evaluating recommendation performance
    
    Preprocessing methods:
    - Do preprocessing before training by adding the following code in the task class:
    >
    def get_before_train_info(self, model):
        if self.fair_controller.get_fairness_opt_type() == "preprocess":
            self.fair_controller.do_preprocess(model)task.
        ...
    
    Inprocess methods:
    - In each batch training, add controller.get_loss() on the model loss before optimizer.step()
    
    PostProcessing methods:
    - Do postprocessing after each epoch by adding the following code in the task class:
    >
    def get_after_epoch_info(self, model):
        if self.fair_controller.get_fairness_opt_type() == "postprocess":
            self.fair_controller.do_postprocess(model):
        ...
    '''
    
    @staticmethod
    def parse_fairness_args(parser):
        '''
        args:
        - fair_lambda
        '''
        parser.add_argument('--fair_lambda', type=float, default=0.1, 
                            help='trade-off coefficient on the fairness loss')
        return parser
    
    def __init__(self, args, reader):
        super().__init__()
        self.fair_lambda = args.fair_lambda
    
    def get_fairness_opt_type(self):
        return "inepoch" # one of {preprocess, inepoch, afterepoch, postprocess}
    
    def log(self):
        print("fairness controller")
        print(f"\tfair_lambda: {self.fair_lambda}")
        print(f"\tfair_opt_type: {self.get_fairness_opt_type()}")
        
    def reset_statistics(self):
        pass
    
    def get_loss(self, model, feed_dict, out_dict, loss):
        pass
    
    def add_fairness_evaluation(self, model, params):
        pass
        
    def do_preprocess(self, model):
        '''
        Pre-processing method for fairness control. E.g. data augmentation for minor groups.
        '''
        pass
    
    def do_after_epoch(self, model, epoch_info):
        '''
        In-process iterative method for fairness control. E.g. non-differentiable fairness contraint optimization.
        '''
        pass
    
    def do_in_epoch(self, model, batch_info):
        '''
        In-process in-epoch method for fairness control. E.g. differentiable fairness contraint optimization.
        '''
        return {}
    
    def do_postprocess(self, model):
        '''
        Post-processing method for fairness control. E.g. data augmentation for minor groups.
        '''
        pass