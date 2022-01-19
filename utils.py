import numpy as np
import csv
from tqdm import tqdm
import os
import sys
import time
import torch
import random

import matplotlib.pyplot as plt
from argparse import Namespace

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

####################################################################
#                           Data Frame                             #
####################################################################

def pd_to_lists(data_frame):
    dataLists = {}
    for key in data_frame:
        dataLists[key] = [eval(data_frame[key].values[i]) for i in range(len(data_frame))]
    return dataLists

####################################################################
#                              Path                                #
####################################################################


def check_folder_exist(fpath):
    if os.path.exists(fpath):
        print("dir \"" + fpath + "\" existed")
    else:
        try:
            os.mkdir(fpath)
        except:
            print("error when creating \"" + fpath + "\"") 
            
def setup_path(fpath, is_dir = True):
    dirs = [p for p in fpath.split("/")]
    curP = ""
    dirs = dirs[:-1] if not is_dir else dirs
    for p in dirs:
        curP += p
        check_folder_exist(curP)
        curP += "/"
        
def get_local_time():
    t = time.localtime()
    return time.strftime("%y/%m/%d, %H:%M:%S", t)
            
#####################################################################
#                              Model                                #
#####################################################################
            
def get_device(model):
    dev = next(model.parameters()).device
    return dev

# def load_and_move_to_cpu(model_path):
#     '''
#     Move GPU model to CPU
#     '''
#     import torch
#     model = torch.load(model_path)
#     model.device = "cpu"
#     model.to("cpu")
#     torch.save(model, model_path)
#     return model
    
# def load_and_move_to_gpu(model_path, device = "cuda:0"):
#     '''
#     Move CPU model to GPU
#     '''
#     import torch
#     model = torch.load(model_path)
#     model.device = device
#     model.cuda()
#     return model

def save_model(model, logger, model_path=None):
    if model_path is None:
        model_path = model.modelPath
    logger.log('Save model to ' + model_path)
    setup_path(model_path)
    torch.save(model.state_dict(), model_path)

# def load_model(logger, model_path):
#     if model_path is None:
#         model_path = self.modelPath
#     logger.log('Load model from ' + model_path)
#     return self.load(model_path)

def count_variables(model) -> int:
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_parameters

######################################################################
#                              Logger                                #
######################################################################

class Logger(object):

    def __init__(self, log_path, on=True):
        self.log_path = log_path
        setup_path(log_path)
        self.on = False
        self.log()
        self.on = on

    def log(self, string = '', newline=True):
        if self.on:
            with open(self.log_path, 'a') as logf:
                logf.write(string)
                if newline: logf.write('\n')

            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()

            
#####################################################################
#                         Result Evaluation                         #
#####################################################################


def get_arg(textline, field):
    return eval(textline[textline.index(field+'='):textline.find(',',textline.index(field+'='),-1)].split('=')[1])

def extract_results(log_root_path, customized_args = [], file_name_identifier = "train_and_eval"):
    result_dict = {}
    for j,file in tqdm(enumerate(os.listdir(log_root_path))):
        if file.endswith(".log") and file_name_identifier in file:
            print(file)
            args = None
            model_name = ""
            results = []
            found = 0
            with open(os.path.join(log_root_path, file), 'r') as fin:
                for i,line in enumerate(fin):
                    if i == 0:
                        model_name = get_arg(line, 'model')
                    if i == 1:
                        args = line.strip()[10:-1]
                    elif "Test set performance" in line:
                        found = 2
                    elif found > 0:
                        if "Result dict" in line:
                            found -= 1
                        elif found == 1:
                            results.append(eval(line))
                            found = 0
            if len(results) > 0:
                args += ','
                result_dict[j] = {'args': args}
                result_dict[j]['model_name'] = model_name
                for k in customized_args:
                    try:
                        result_dict[j][k] = get_arg(args, k)
                    except:
                        result_dict[j][k] = 'NaN'
                results = {k:[result[k] for result in results] for k in results[0].keys()}
                for k,v in results.items():
                    result_dict[j][k] = v
    return result_dict

def read_line_number(file_path, line_num):
    with open(file_path, 'r') as fin:
        for i,line in enumerate(fin):
            if i == line_num:
                return line.strip()
    return ""

def extract_args(log_path):
    print(log_path)
    argstr = read_line_number(log_path, 1)
    if 'fair_lambda' not in argstr:
        argstr = argstr[:-1] + ', fair_lambda=0.1)'
    if 'fair_rho' not in argstr:
        argstr = argstr[:-1] + ', fair_rho=1)'
    if 'fair_group_feature' not in argstr:
        argstr = argstr[:-1] + f", fair_group_feature='{group_feature}')"
    args = eval(argstr)
    return args

def extract_epochwise_result(log_path, keyword = 'Result dict:', next_line = True):
    '''
    @output:
    - round_result: [epoch_result]
        - epoch_result: {metric: value}
    '''
    round_result = []
    wait_flag, read_flag = True, False
    with open(log_path, 'r') as fin:
        for i,line in enumerate(fin):
            if 'Epoch ' in line:
                wait_flag = True # wait for keyward
            elif keyword == line[:len(keyword)]:
                if wait_flag: # read next line
                    read_flag = True
                    wait_flag = False
                    if not next_line:
                        read_flag = False
                        round_result.append(eval(line.strip()[len(keyword):]))
                else: # not training process, stop and save
                    break
            elif read_flag:
                read_flag = False
                round_result.append(eval(line))
    return round_result

####################################################################
#                               Plot                               #
####################################################################


def plot_ordinal_statistics(stats, features, ncol = 3):
    '''
    @input:
    - stats: {field_name: {key: [values]}}
    - features: [field_name]
    - ncol: number of subplots in each row
    '''
    assert ncol > 0
    N = len(features)
    plt.figure(figsize = (16, 4*((N-1)//ncol+1)))
    for i,field in enumerate(features):
        plt.subplot((N-1)//ncol+1,ncol,i+1)
        field_stats = stats[field] # {key: [values]}
        X = sorted(list(field_stats.keys()))
        Y = [np.mean(field_stats[x]) for x in X]
        plt.bar(X,Y)
        plt.title(field)
        scale = 1e-7 + np.max(Y) - np.min(Y)
        plt.ylim(np.min(Y) - scale * 0.05, np.max(Y) + scale * 0.05)
    plt.show()

def plot_multiple_line(stats, features, ncol = 2, row_height = 4,
                       ylabel = 'y', xlabel = 'x', legend_title = ''):
    '''
    @input:
    - stats: {field_name: {key: [values]}}
    - features: [field_name]
    - ncol: number of subplots in each row
    '''
    assert ncol > 0
    N = len(features)
    fig_height = 12 // ncol if len(features) == 1 else row_height*((N-1)//ncol+1)
    plt.figure(figsize = (16, fig_height))
    for i,field in enumerate(features):
        plt.subplot((N-1)//ncol+1,ncol,i+1)
        minY,maxY = float('inf'),float('-inf')
        for key, value_list in stats[field].items():
#             print(key, value_list)
            X = np.arange(1,len(value_list)+1)
            minY,maxY = min(minY,min(value_list)),max(maxY,max(value_list))
            plt.plot(X,value_list,label = key)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(field)
        scale = 1e-4 + maxY - minY
        plt.ylim(minY - scale * 0.05, maxY + scale * 0.05)
        plt.legend(title = legend_title)
    plt.show()
    
    
def plot_recommendation_over_lambda(stats, lambdas, metrics, other_model_results = {}, 
                                    row_height = 4, ncol = 3, legend_appear_at = 0, colors = {}):
    '''
    @input:
    - stats: {fair_model_name: {metric: {lambda: [values]}}}
    - other_model_results: {metric: value}
    - features: [field_name]
    - ncol: number of subplots in each row
    '''
    assert ncol > 0
    N = len(metrics)
    X = lambdas
    fig_height = 12 // ncol if len(metrics) == 1 else row_height*((N-1)//ncol+1)
    plt.figure(figsize = (16, fig_height))
    for i,field in enumerate(metrics):
        plt.subplot((N-1)//ncol+1,ncol,i+1)
        minY, maxY = float('inf'), float('-inf')
        for fair_model_name, model_stats in stats.items():
            Y = np.array(model_stats[field])
            minY, maxY = min(minY, min(Y)), max(maxY, max(Y))
            if legend_appear_at == i:
                c = colors[fair_model_name] if fair_model_name in colors else '#ababab'
                plt.plot(X, Y, label = fair_model_name, color = c)
            else:
                c = colors[fair_model_name] if fair_model_name in colors else '#ababab'
                plt.plot(X, Y, color = c)
        for other_model_name, model_stats in other_model_results.items():
            Y = np.array([model_stats[field]] * len(X))
            minY, maxY = min(minY, min(Y)), max(maxY, max(Y))
            if legend_appear_at == i:
                c = colors[other_model_name] if other_model_name in colors else '#ababab'
                plt.plot(X, Y, ':', label = other_model_name, color = c)
            else:
                c = colors[other_model_name] if other_model_name in colors else '#ababab'
                plt.plot(X, Y, ':', color = c)
        plt.title(field)
#         plt.xticks(X)
        if legend_appear_at == i:
            plt.legend()
        scale = 1e-7 + maxY - minY
        plt.ylim(minY - scale * 0.05, maxY + scale * 0.05)
    plt.show()
