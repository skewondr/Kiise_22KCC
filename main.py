from config import ARGS
from util import (
    get_data_user_sep
)
from dataset.dataset_user_sep import UserSepDataset
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


def run(i):
    """
    i: single integer represents dataset number
    """
    logger.info(f"Dataset: {ARGS.dataset_name}")
    data_path = f'../dataset/{ARGS.dataset_name}/processed'

    train_sample_data, num_of_train_user = get_data_user_sep(data_path, i, 'train')
    val_sample_data, num_of_val_user = get_data_user_sep(data_path, i, 'val')
    test_sample_data, num_of_test_user = get_data_user_sep(data_path, i, 'test')
    #import IPython; IPython.embed(); exit(1);

    train_data = UserSepDataset('train', train_sample_data, ARGS.dataset_name)
    val_data = UserSepDataset('val', val_sample_data, ARGS.dataset_name)
    test_data = UserSepDataset('test', test_sample_data, ARGS.dataset_name)

    logger.info(f'Train: # of users: {num_of_train_user}, # of samples: {len(train_sample_data)}')
    logger.info(f'Validation: # of users: {num_of_val_user}, # of samples: {len(val_sample_data)}')
    logger.info(f'Test: # of users: {num_of_test_user}, # of samples: {len(test_sample_data)}')
    
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

    trainer = Trainer(model, ARGS.device, ARGS.warm_up_step_count,
                      d_model, ARGS.num_epochs, ARGS.weight_path,
                      ARGS.lr, train_data, val_data, test_data)
    start_time = time.time()
    logger.info("Training")
    trainer.train()
    total_time = time.time() - start_time
    logger.info(f"elapsed time: {total_time:.2f}s, {timedelta(seconds=total_time)}")
    logger.info("Evaluation")
    trainer.test(0)
    return trainer.test_acc, trainer.test_auc


if __name__ == '__main__':
    
    logger.info(f"seed: {ARGS.random_seed}")
    set_seed(ARGS.random_seed)
    
    if not ARGS.test_run:
        
        now = datetime.now()
        now_str = f'{now.day:02}{now.hour:02}{now.minute:02}'
        
        log_filename = now_str+".log"
        logfile_path = os.path.join(ARGS.ckpt_path, log_filename) #./checkpoint/{ckpt_name}/{now}.log 
        if os.path.exists(logfile_path) and not ARGS.resume and not ARGS.test_only:
            os.remove(logfile_path)

        set_logger(os.path.join(ARGS.ckpt_path, log_filename))
        if ARGS.run_script is not None:
            copy_file(#copy .sh 
                ARGS.run_script, os.path.join(ARGS.ckpt_path, os.path.basename(ARGS.run_script))
            )
    
    if ARGS.cross_validation is False:
        test_acc, test_auc = run(1)
    else:
        acc_list = []
        auc_list = []

        for i in range(1, 6):
            logger.info(f'{i}th dataset')
            test_acc, test_auc = run(i)
            acc_list.append(test_acc)
            auc_list.append(test_auc)

        acc_array = np.asarray(acc_list)
        auc_array = np.asarray(auc_list)
        logger.info(f'mean acc: {np.mean(acc_array):.4f}, auc: {np.mean(auc_array):.4f}')
        logger.info(f'std acc: {np.std(acc_array):.4f}, auc: {np.std(auc_array):.4f}')
