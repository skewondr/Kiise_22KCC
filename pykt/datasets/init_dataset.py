import os, sys

from torch.utils.data import DataLoader
import numpy as np
from .data_loader import KTDataset
from .dkt_forget_dataloader import DktForgetDataset
from IPython import embed

def init_test_datasets(data_config, model_name, batch_size):
    test_question_loader, test_question_window_loader = None, None
    if model_name in ["dkt_forget"]:
        test_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config, {-1})
        test_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
                                        data_config, {-1})
        if "test_question_file" in data_config:
            test_question_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config, {-1}, True)
            test_question_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config, {-1}, True)
    else:

        test_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config, {-1})
        test_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config, {-1})
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_file"]), data_config, {-1}, True)
            test_question_window_dataset = KTDataset(os.path.join(data_config["dpath"], data_config["test_question_window_file"]), data_config, {-1}, True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    if "test_question_file" in data_config:
        print(f"has test_question_file!")
        test_question_loader = DataLoader(test_question_dataset, batch_size=batch_size, shuffle=False)
        test_question_window_loader = DataLoader(test_question_window_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, test_window_loader, test_question_loader, test_question_window_loader

def update_gap(max_rgap, max_sgap, max_pcount, cur):
    max_rgap = cur.max_rgap if cur.max_rgap > max_rgap else max_rgap
    max_sgap = cur.max_sgap if cur.max_sgap > max_sgap else max_sgap
    max_pcount = cur.max_pcount if cur.max_pcount > max_pcount else max_pcount
    return max_rgap, max_sgap, max_pcount

def init_dataset4train(device, emb_type, dataset_name, model_name, data_config, i, batch_size):
    data_config = data_config[dataset_name]
    all_folds = set(data_config["folds"])
    if model_name == "dkt_forget":
        max_rgap, max_sgap, max_pcount = 0, 0, 0
        curvalid = DktForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config, {i})
        curtrain = DktForgetDataset(os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config, all_folds - {i})
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curtrain)
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, curvalid)
    else:
        curvalid = KTDataset(device, os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config, {i})
        curtrain = KTDataset(device, os.path.join(data_config["dpath"], data_config["train_valid_file"]), data_config, all_folds - {i})

    # if emb_type.startswith("R"):
    curtrain.emb_type = emb_type
    curvalid.emb_type = emb_type

    n_tokens = int(emb_type.split("_")[-1]) if emb_type.startswith(("R_", "L_")) else 2
    print("n_tokens",n_tokens)
    curtrain.get_quantiles(n_tokens)
    curvalid.get_quantiles(n_tokens)

    train_loader = DataLoader(curtrain, batch_size=batch_size)
    valid_loader = DataLoader(curvalid, batch_size=batch_size)
    
    if model_name == "dkt_forget":
        test_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_file"]), data_config, {-1})
        test_window_dataset = DktForgetDataset(os.path.join(data_config["dpath"], data_config["test_window_file"]),
                                        data_config, {-1})
        max_rgap, max_sgap, max_pcount = update_gap(max_rgap, max_sgap, max_pcount, test_dataset)
    else:
        test_dataset = KTDataset(device, os.path.join(data_config["dpath"], data_config["test_file"]), data_config, {-1})
        test_window_dataset = KTDataset(device, os.path.join(data_config["dpath"], data_config["test_window_file"]), data_config, {-1})
    
    if model_name == "dkt_forget":
        data_config["num_rgap"] = max_rgap + 1
        data_config["num_sgap"] = max_sgap + 1
        data_config["num_pcount"] = max_pcount + 1

    # if emb_type.startswith("R"):
    test_dataset.emb_type = emb_type
    n_tokens = int(emb_type.split("_")[-1]) if emb_type.startswith(("R_", "L_")) else 2
    print("n_tokens",n_tokens)
    test_dataset.get_quantiles(n_tokens)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader = None

    return train_loader, valid_loader, test_loader, test_window_loader