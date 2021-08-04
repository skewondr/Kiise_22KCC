from config import ARGS
from util import (
    get_data_user_sep
)
from dataset.dataset_user_sep import UserSepDataset
from util import load_checkpoint
from network.DKT import DKT
from network.DKVMN import DKVMN
from network.NPA import NPA
from network.SAKT import SAKT
from constant import QUESTION_NUM
from trainer import Trainer
import numpy as np
import time
from datetime import timedelta, datetime
import torch
import torch.nn as nn
import os
import logzero
from logzero import logger
from functools import wraps
import shutil
import random 
from torch.optim.lr_scheduler import CosineAnnealingLR
from optimizers import DenseSparseAdam

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True

def copy_file(src: str, dst: str) -> None:
    try:
        shutil.copyfile(src, dst)
    except shutil.SameFileError:
        pass

def log_elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()

        elapsed = end - start
        logger.info(f"elapsed time: {end - start:.2f}s, {timedelta(seconds=elapsed)}")

        return ret

    return wrapper

def set_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # start writing into a logfile 
    logzero.logfile(log_path)

def get_optimizer(model_name: str, network: nn.Module, lr: float, decay: float):
    if model_name in TRANSFORMER_MODELS:
        no_decay = ["bias", "LayerNorm.weight"]
        param_groups = [
            {
                "params": [
                    p
                    for n, p in network.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": decay,
            },
            {
                "params": [
                    p
                    for n, p in network.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = DenseSparseAdam(param_groups, lr=lr)
    else:
        optimizer = DenseSparseAdam(network.parameters(), lr=lr, weight_decay=decay)

    return optimizer

TRANSFORMER_MODELS = ["SAKT"]

def get_model():
    if ARGS.model == 'DKT':
        model = DKT(ARGS.input_dim, ARGS.hidden_dim, ARGS.num_layers, QUESTION_NUM[ARGS.dataset_name],
                    ARGS.dropout).to(ARGS.device)
        d_model = ARGS.hidden_dim

    elif ARGS.model == 'DKVMN':
        model = DKVMN(ARGS.key_dim, ARGS.value_dim, ARGS.summary_dim, QUESTION_NUM[ARGS.dataset_name],
                      ARGS.concept_num).to(ARGS.device)
        d_model = ARGS.value_dim

    elif ARGS.model == 'NPA':
        model = NPA(ARGS.input_dim, ARGS.hidden_dim, ARGS.attention_dim, ARGS.fc_dim,
                    ARGS.num_layers, QUESTION_NUM[ARGS.dataset_name], ARGS.dropout).to(ARGS.device)
        d_model = ARGS.hidden_dim

    elif ARGS.model == 'SAKT':
        model = SAKT(ARGS.hidden_dim, QUESTION_NUM[ARGS.dataset_name], ARGS.num_layers,
                     ARGS.num_head, ARGS.dropout).to(ARGS.device)
        d_model = ARGS.hidden_dim

    else:
        raise NotImplementedError

    return model, d_model

def run(i, model, start_epoch, optimizer, scheduler, other_states):
    """
    i: single integer represents dataset number
    """
    ################################## Prepare Dataset ###############################
    data_path = f'../dataset/{ARGS.dataset_name}/processed'

    train_sample_data, num_of_train_user = get_data_user_sep(data_path, i, 'train')
    val_sample_data, num_of_val_user = get_data_user_sep(data_path, i, 'val')
    test_sample_data, num_of_test_user = get_data_user_sep(data_path, i, 'test')
    #import IPython; IPython.embed(); exit(1);

    train_data = UserSepDataset('train', train_sample_data, ARGS.dataset_name)
    val_data = UserSepDataset('val', val_sample_data, ARGS.dataset_name)
    test_data = UserSepDataset('test', test_sample_data, ARGS.dataset_name)

    logger.info(f"Dataset: {ARGS.dataset_name}")
    logger.info(f'Train: # of users: {num_of_train_user}, # of samples: {len(train_sample_data)}')
    logger.info(f'Validation: # of users: {num_of_val_user}, # of samples: {len(val_sample_data)}')
    logger.info(f'Test: # of users: {num_of_test_user}, # of samples: {len(test_sample_data)}')
    
    trainer = Trainer(
        model, 
        ARGS.test_run,
        ARGS.device,
        ARGS.eval_steps,
        ARGS.num_epochs, 
        start_epoch,
        optimizer,
        scheduler,
        ARGS.ckpt_path,
        ARGS.es_patience, 
        train_data, 
        val_data, 
        test_data,
        other_states=other_states,
    )
    if ARGS.mode == "train":
        start_time = time.time()
        logger.info("Training")
        trainer.train()
        total_time = time.time() - start_time
        logger.info(f"elapsed time: {total_time:.2f}s, {timedelta(seconds=total_time)}")
    if ARGS.mode == "eval":
        now = datetime.now()
        now_str = f'{now.day:02}{now.hour:02}{now.minute:02}'
        copy_file(#copy .pt
                os.path.join(ARGS.ckpt_path, 'best_ckpt.pt'), os.path.join(ARGS.weight_path, f'{now_str}.pt')
            )
    trainer.test()
    return trainer.test_acc, trainer.test_auc

if __name__ == '__main__':
    
    log_filename = "logger.log"
    logfile_path = os.path.join(ARGS.ckpt_path, log_filename)
    if not ARGS.test_run: 
        set_logger(logfile_path)
    logger.info(f"seed: {ARGS.random_seed}")
    set_seed(ARGS.random_seed)
    ################################# Prepare Model ##################################
    logger.info(f"Model: {ARGS.model}")
    model, d_model = get_model()
    if torch.cuda.is_available():
        ARGS.device = 'cuda'
        num_gpus = torch.cuda.device_count()
        if ARGS.gpu != 'none':
            os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu
            logger.info("Single-GPU mode")
            torch.cuda.set_device(int(ARGS.gpu))
        elif num_gpus > 1 :
            logger.info(f"Multi-GPU mode: {num_gpus} GPUs")
            model = nn.DataParallel(model)
        else:
            logger.info("CPU mode")
    ################################### Training #####################################
    optimizer = get_optimizer(ARGS.model, model, ARGS.lr, ARGS.decay)
    if ARGS.eta_min is not None:
        #lr이 eta_min까지 떨어졌다 다시 초기 lr까지 올라온다. 
        #lr은 T_max*2 단위로 반복한다. 
        scheduler = CosineAnnealingLR(
            optimizer, T_max=max(3, ARGS.num_epochs // 10), eta_min=ARGS.eta_min
        )
    else:
        scheduler = None   
    if ARGS.mode == "train":
        
        if ARGS.run_script is not None:
            copy_file(#copy .sh 
                ARGS.run_script, os.path.join(ARGS.ckpt_path, os.path.basename(ARGS.run_script))
            )
        if ARGS.ckpt_resume:
            logger.info(f"Resume Training")
            start_epoch, other_states = load_checkpoint(
                    os.path.join(ARGS.ckpt_path, "last_ckpt.pt"), model, optimizer, scheduler, return_other_states=True
                )
        else:
            start_epoch = 0
            other_states = {}
        ################################### Start Training #####################################
        if ARGS.cross_validation is False:
            test_acc, test_auc = run(1, model, start_epoch, optimizer, scheduler, other_states)
        else:
            acc_list = []
            auc_list = []

            for i in range(1, 6):
                logger.info(f'{i}th dataset')
                test_acc, test_auc = run(i, model, start_epoch, optimizer, scheduler, other_states)
                acc_list.append(test_acc)
                auc_list.append(test_auc)

            acc_array = np.asarray(acc_list)
            auc_array = np.asarray(auc_list)
            logger.info(f'mean acc: {np.mean(acc_array):.4f}, auc: {np.mean(auc_array):.4f}')
            logger.info(f'std acc: {np.std(acc_array):.4f}, auc: {np.std(auc_array):.4f}')

    if ARGS.mode == "eval":
        model.load_state_dict(torch.load(os.path.join(ARGS.ckpt_path, "best_ckpt.pt")))
        start_epoch = 0
        other_states = {}
        test_acc, test_auc = run(1, model, start_epoch, optimizer, scheduler, other_states)
        