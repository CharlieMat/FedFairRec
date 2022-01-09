import os
import time
import argparse
import setproctitle

import numpy as np
import torch

import utils


#################################################################################
#                              Command Interface                                #
#################################################################################

from model.baselines import *
from model.fed_rec import *
from reader import *
from task import *

if __name__ == '__main__':
    
    # initial args
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--proctitle', type=str, default='Elizabeth Bennet', help='process title on CLT')
    init_parser.add_argument('--model', type=str, default='MF', help='Create a model to run.')
    init_parser.add_argument('--task', type=str, default='TopK', help='Task to run')
    initial_args, _ = init_parser.parse_known_args()
    print(initial_args)
    modelClass = eval('{0}.{0}'.format(initial_args.model))
    taskClass = eval('{0}.{0}'.format(initial_args.task))
    readerClass = eval('{0}.{0}'.format(modelClass.get_reader()))
#     setproctitle.setproctitle(initial_args.proctitle+"("+initial_args.model+"-"+initial_args.task+")")
    setproctitle.setproctitle(initial_args.proctitle)

    # control args
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=-1,
                        help='-1 if using cpu; 0,1... if using cuda')
    parser.add_argument('--seed', type=int, default=9,
                        help='random seed')
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Run train")
    mode_group.add_argument("--train_and_eval", action="store_true", help="Run train")
    mode_group.add_argument("--continuous_train", action="store_true", help="Run continous train")
    mode_group.add_argument("--eval", action="store_true", help="Run eval")
    
    # customized args
    parser = modelClass.parse_model_args(parser)
    parser = readerClass.parse_data_args(parser)
    parser = taskClass.parse_task_args(parser)
    args, _ = parser.parse_known_args()
    print(args)
    
    # reproducibility
    utils.set_random_seed(args.seed)
    
    # GPU
    if args.cuda >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        torch.cuda.set_device(args.cuda)
        device = "cuda:" + str(args.cuda)
    else:
        device = "cpu"
    
    # reader
    print("Setup reader")
    reader = readerClass(args)
    print(reader.get_statistics())
    
    
    # task
    print("Setup task")
    task = taskClass(args, reader)
    
    # run task
    if args.train or args.train_and_eval:
        # train model
        for i in range(task.n_round):
            print(f"#######################\r\n#       Round {i+1}       #\r\n#######################")
            # model
            model = modelClass(args, reader, device)
            model.log()
            task.log()
            model.show_params()
            model = model.to(device)
            task.train(model, continuous = False)
            if args.train_and_eval:
                task.do_test(model)
    else:
        model = modelClass(args, reader, device)
        model.to(device)
        if args.continuous_train:
            task.train(model, continuous = True)
        task.do_test(model)
    
    
    