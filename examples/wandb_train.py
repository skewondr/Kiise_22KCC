import os
import argparse
import json

import torch
torch.set_num_threads(4) 
from torch.optim import SGD, Adam
import copy

from pykt.models import train_model,evaluate,init_model
from pykt.utils import debug_print,set_seed
from pykt.datasets import init_dataset4train

import time
from datetime import timedelta, datetime
import numpy as np
import time
from time import localtime 
from IPython import embed
import wandb

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# device = "cpu" if not torch.cuda.is_available() else "cuda"
# os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

def save_config(train_config, model_config, data_config, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)

class EarlyStopping:
    #https://stats.stackexchange.com/questions/68893/area-under-curve-of-roc-vs-overall-accuracy
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(
        self, 
        val_auc, 
        epoch=None, 
    ):

        score = val_auc

        if self.best_score is None:
            self.best_score = score
        
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
def main(train_params, model_params):

    os.environ["CUDA_VISIBLE_DEVICES"] = train_params['gpu']
    torch.cuda.set_device(int(train_params['gpu']))
    device = "cpu" if not torch.cuda.is_available() else f"cuda:{train_params['gpu']}"

    tst_acc_list = []
    tst_auc_list = []
    val_acc_list = []
    val_auc_list = []
    ckpt_name_list = ['dataset_name', 'model_name', 'seed', 'fold']
    params_str = "_".join([str(train_params[_]) for _ in ckpt_name_list]) 
    tm = localtime(time.time())
    params_str += f'_{tm.tm_mday}{tm.tm_hour}{tm.tm_min}{tm.tm_sec}'
    total_time = ""

    for i in range(train_params['fold']):
        if "use_wandb" not in model_params:
            model_params['use_wandb'] = 1

        if model_params['use_wandb']==1:
            wandb.init(project="pykt-yj-examples", entity="pykt-framework")
            wandb.run.name = params_str
            wandb.run.save()

        set_seed(train_params["seed"])
        model_name, dataset_name, fold, emb_type, save_dir = train_params["model_name"], train_params["dataset_name"], \
            i, train_params["emb_type"], train_params["save_dir"]
            
        debug_print(text = "load config files.",fuc_name="main")

        with open("../configs/kt_config_yj.json") as f:
            config = json.load(f)
            train_config = config["train_config"]
            model_config = config[model_name][dataset_name]
            for k, v in model_params.items():
                if k not in model_config:
                    model_config[k] = v

        batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]
        seq_len = train_config["seq_len"]

        with open("../configs/data_config.json") as fin:
            data_config = json.load(fin)
        print("Start init data")
        # print(dataset_name, model_name, data_config, fold, batch_size)
        
        debug_print(text="init_dataset",fuc_name="main")
        train_loader, valid_loader, test_loader, test_window_loader = init_dataset4train(device, dataset_name, model_name, data_config, fold, batch_size)

        if model_config['add_uuid'] == 1:
            import uuid
        ckpt_path = os.path.join(save_dir, params_str)
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)
        print(f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}")
        print(f"model_config: {model_config}")
        print(f"train_config: {train_config}")

        save_config(train_config, model_config, data_config[dataset_name], ckpt_path)
        learning_rate = model_config["learning_rate"]
        for remove_item in ['use_wandb','learning_rate','add_uuid']:
            if remove_item in model_config:
                del model_config[remove_item]
        if model_name in ["saint", "sakt"]:
            model_config["seq_len"] = seq_len
            
        debug_print(text = "init_model",fuc_name="main")
        model = init_model(device, model_name, model_config, data_config[dataset_name], emb_type)

        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)
    
        testauc, testacc = -1, -1
        window_testauc, window_testacc = -1, -1
        validauc, validacc = -1, -1
        best_epoch = -1
        save_model = True
        
        debug_print(text = "train model",fuc_name="main")
        
        early_stopping = EarlyStopping(patience=train_config["es_patience"], verbose=True, path=ckpt_path)   
        start_time = time.time()
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(device, model, train_loader, valid_loader, num_epochs, opt, ckpt_path, early_stopping, test_loader, test_window_loader, save_model)
        
        val_acc_list.append(validacc)
        val_auc_list.append(validauc)

        if save_model:
            best_model = init_model(device, model_name, model_config, data_config[dataset_name], emb_type)
            net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
            best_model.load_state_dict(net)
            # evaluate test
            
            if test_loader != None:
                save_test_path = os.path.join(ckpt_path, emb_type+"_test_predictions.txt")
                testauc, testacc = evaluate(device, best_model, test_loader, model_name)#, save_test_path)
            if test_window_loader != None:
                save_test_path = os.path.join(ckpt_path, emb_type+"_test_window_predictions.txt")
                window_testauc, window_testacc = evaluate(device, best_model, test_window_loader, model_name)#, save_test_path)
            # window_testauc, window_testacc = -1, -1
            # trainauc, trainacc = self.evaluate(train_loader, emb_type)
            testauc, testacc, window_testauc, window_testacc = round(testauc, 4), round(testacc, 4), round(window_testauc, 4), round(window_testacc, 4)
            tst_acc_list.append(testacc)
            tst_auc_list.append(testauc)

        print('-'*80)
        print("fold\tmodelname\tembtype\ttestauc\ttestacc\tvalidauc\tvalidacc\tbest_epoch")
        print(f"{str(fold)}\t{model_name}\t\t{emb_type}\t{str(testauc)}\t{str(testacc)}\t{str(validauc)}\t{str(validacc)}\t{str(best_epoch)}")
        print('-'*80)
        model_save_path = os.path.join(ckpt_path, emb_type+"_model.ckpt")
        total_time = str(timedelta(seconds=time.time() - start_time))
        print(f"elapsed time: {total_time}")
        
        # if model_params['use_wandb']==1:
        #     wandb.log({"testauc": testauc, "testacc": testacc, "window_testauc": window_testauc, "window_testacc": window_testacc, 
        #                 "validauc": validauc, "validacc": validacc, "best_epoch": best_epoch,"model_save_path":model_save_path})

    print_result = {
            "dataset":train_config['dataset_name'],
            "model":train_config['model_name'],
            "emb type":train_config['emb_type'],
            "seed":train_config['seed'],
            "kfolds":train_config['fold'],
            "elapsed time": total_time,
            "mean testauc": np.array(tst_auc_list).mean(),
            "mean testauc": np.array(tst_auc_list).mean(),
            "mean testacc": np.array(tst_acc_list).mean(),
            "mean validauc": np.array(val_auc_list).mean(),
            "mean validacc": np.array(val_acc_list).mean(),
            "best_fold(auc)": tst_auc_list.index(max(tst_auc_list))}
    print(print_result)
    if model_config['use_wandb']==1:
        wandb.log(print_result)
