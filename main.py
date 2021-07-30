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
   

    user_base_path = f'../dataset/{ARGS.dataset_name}/processed'

    train_sample_data, num_of_train_user = get_data_user_sep(user_base_path, i, 'train')
    val_sample_data, num_of_val_user = get_data_user_sep(user_base_path, i, 'val')
    test_sample_data, num_of_test_user = get_data_user_sep(user_base_path, i, 'test')
    #import IPython; IPython.embed(); exit(1);

    train_data = UserSepDataset('train', train_sample_data, ARGS.dataset_name)
    val_data = UserSepDataset('val', val_sample_data, ARGS.dataset_name)
    test_data = UserSepDataset('test', test_sample_data, ARGS.dataset_name)

    print(f'Train: # of users: {num_of_train_user}, # of samples: {len(train_sample_data["data_path"])}')
    print(f'Validation: # of users: {num_of_val_user}, # of samples: {len(val_sample_data["data_path"])}')
    print(f'Test: # of users: {num_of_test_user}, # of samples: {len(test_sample_data["data_path"])}')
    
    model, d_model = get_model()

    trainer = Trainer(model, ARGS.device, ARGS.warm_up_step_count,
                      d_model, ARGS.num_epochs, ARGS.weight_path,
                      ARGS.lr, train_data, val_data, test_data)
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time
    print(f'[END]     time: {total_time:.2f}')
    trainer.test(0)
    return trainer.test_acc, trainer.test_auc


if __name__ == '__main__':
    

    if ARGS.cross_validation is False:
        test_acc, test_auc = run(1)
    else:
        acc_list = []
        auc_list = []

        for i in range(1, 6):
            print(f'{i}th dataset')
            test_acc, test_auc = run(i)
            acc_list.append(test_acc)
            auc_list.append(test_auc)

        acc_array = np.asarray(acc_list)
        auc_array = np.asarray(auc_list)
        print(f'mean acc: {np.mean(acc_array):.4f}, auc: {np.mean(auc_array):.4f}')
        print(f'std acc: {np.std(acc_array):.4f}, auc: {np.std(auc_array):.4f}')
