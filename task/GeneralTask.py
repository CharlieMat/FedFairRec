import os
import torch
import numpy as np
from time import time

import utils

class GeneralTask(object):
    @staticmethod
    def parse_task_args(parser):
        parser.add_argument('--n_round', type=int, default=5, 
                            help='number of rounds of experiment')
        parser.add_argument('--optimizer', type=str, default='adam',
                            help='optimizer type')
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check some tensors every check_epoch.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=1,
                            help='Batch size during testing.')
        parser.add_argument('--temper', type=int, default=5,
                            help='default temper 5')
        parser.add_argument('--with_val', action='store_true', 
                            help='converge on val set')
        parser.add_argument('--val_sample_p', type=float, default=1.0, 
                            help='do validation on a proportion of the data')
        parser.add_argument('--test_sample_p', type=float, default=1.0, 
                            help='do test on a proportion of the data')
        parser.add_argument('--stop_metric', type=str, default='metric', 
                            help='the evaluation metric for stop criteria')
        parser.add_argument('--pin_memory', action='store_true',
                            help='pin_memory in DataLoader')
        return parser


    def __init__(self, args, reader):
        self.n_round = args.n_round
        self.optimizer = args.optimizer
        self.epoch = np.float("inf") if args.epoch < 0 else args.epoch
        self.check_epoch = args.check_epoch
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.with_val = args.with_val
        self.val_sample_p = args.val_sample_p
        self.test_sample_p = args.test_sample_p
        self.stop_metric = args.stop_metric[1:] if "_" in args.stop_metric else args.stop_metric
        self.stop_metric_sign = -1. if "_" in args.stop_metric else 1.
        self.pin_memory = args.pin_memory
        self.init_temper = args.temper
    
    def log(self):
        print("Task params")
        print(f"\tn_round: {self.n_round}")
        print(f"\toptimizer: {self.optimizer}")
        print(f"\tepoch: {self.epoch}")
        print(f"\tcheck_epoch: {self.check_epoch}")
        print(f"\tlr: {self.lr}")
        print(f"\tbatch_size: {self.batch_size}")
        print(f"\teval_batch_size: {self.eval_batch_size}")
        print(f"\twith_val: {self.with_val}")
        print(f"\tval_sample_p: {self.val_sample_p}")
        print(f"\ttest_sample_p: {self.test_sample_p}")
        print(f"\tstop_metric: {self.stop_metric}")
        print(f"\tstopMetricSign: {self.stop_metric_sign}")
        print(f"\tpin_memory: {self.pin_memory}")
        print(f"\tinit_temper: {self.init_temper}")
    
    def train(self, model, continuous = False):
        print("Training")
        # optimizer
        self._build_optimizer(model)
        if continuous:
            model.load_from_checkpoint(model.model_path)
        
        # preparation before training
        model.actions_before_train(self.get_before_train_info(model))
        model.reader.set_phase("train")
            
        # optimization
        best_val_loss = 1.
#         null_report = self.do_val(model)
#         test_null_report = self.do_test(model, reload = False)
#         best_val_loss = self.stop_metric_sign * null_report[self.stop_metric]
#         print(f"Before train performance: \n\tval:{null_report[self.stop_metric]}\n\ttest:{test_null_report[self.stop_metric]}")
#         print(f"Before train performance: \n\ttest:{test_null_report[self.stop_metric]}")
        
        temper = 3
        try:
            epo = 0
            while epo < self.epoch:
                epo += 1
                # train an epoch
                model.actions_before_epoch(self.get_before_epoch_info(model))
                model.train()
                t1 = time()
                train_report = self.do_epoch(model, epo)
                print("Epoch {}; time {:.4f}; loss: {:.4f}".format(epo, time() - t1, train_report["loss"]))
                model.actions_after_epoch(self.get_after_epoch_info(model))

                # check validation and test set
                if epo % self.check_epoch == 0:
                    if self.with_val:
                        t2 = time()
                        val_report = self.do_val(model)
                        print("\t validation - time {}; metric: {:.4f}".format(time() - t2, val_report[self.stop_metric]))

                        # save best model and early termination
                        metric = self.stop_metric_sign * val_report[self.stop_metric]
                        if epo == 1 or metric < best_val_loss - 1e-3:
                            model.save_checkpoint()
                            temper = self.init_temper
                            best_val_loss = metric
                            print("Save model and reset temper to " + str(self.init_temper))
                        else:
                            temper -= 1
                            print("Temper down to " + str(temper))
                            if temper == 0:
                                print("Out of temper, early termination.")
                                break
        except KeyboardInterrupt:
            print("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                print(os.linesep + '-' * 20 + ' END: ' + utils.get_local_time() + ' ' + '-' * 20)
                exit(1)
        model.actions_after_train(self.get_after_train_info(model))
        
    def _build_optimizer(self, model):
        opt_type = self.optimizer.lower()
        if opt_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        elif opt_type == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.lr)
        elif opt_type == 'adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), lr=self.lr)
        elif opt_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        else:
            raise ValueError("Unknown Optimizer: " + self.optimizer)
        model.optimizer = optimizer
    
    def show_batch(self, batch):
        for k, batch in batch.items():
            if torch.is_tensor(batch):
                print(f"{k}: size {batch.shape}, \n\tfirst 5 {batch[:5]}")
            else:
                print(f"{k}: {batch}")
    
    def do_val(self, model):
        self.eval_sample_p = self.val_sample_p
        model.reader.set_phase("val")
        return self.do_eval(model)
    
    def do_test(self, model, reload = True):
        print("Test set performance")
        self.eval_sample_p = self.test_sample_p
        # load the best model saved for test
        if reload:
            model.load_from_checkpoint(model.model_path, with_optimizer = False)
        model.to(model.device)
        model.reader.set_phase("test")
        return self.do_eval(model)
    
    ############################################
    #          Require Implementation          #
    ############################################
    
    def get_before_train_info(self, model): 
        pass
    
    def get_before_epoch_info(self, model):
        pass
    
    def get_after_train_info(self, model): 
        pass
    
    def get_after_epoch_info(self, model):
        pass
    
    def do_epoch(self, model, epoch_id):
        pass
    
    def do_eval(self, model):
        pass
    