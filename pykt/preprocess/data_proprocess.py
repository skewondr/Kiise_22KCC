import os, sys
from .split_datasets import main as split

def process_raw_data(dataset_name,dname2paths):
    readf = dname2paths[dataset_name]
    assist2015_list = ["assist2015", "assist2015_q2a_k5", "assist2015_q2a_k25", "assist2015_q2a_k50"]
    if dataset_name == "ednet": 
        dname = "/".join(readf.split("/")[0:-2]) + "/sub1_fold3"
    elif dataset_name in assist2015_list: 
        if dataset_name == "assist2015":
            dname = "/".join(readf.split("/")[0:-1]) + "/fold3"
        else:
            q2a_name = "_".join(dataset_name.split("_")[-2:])
            dname = "/".join(readf.split("/")[0:-1]) + f"/{q2a_name}_fold3"
    else: 
        dname = "/".join(readf.split("/")[0:-1])
    writef = os.path.join(dname, "data.txt")
    print(f"Start preprocessing data: {dataset_name}")
    if dataset_name == "assist2009":
        from .assist2009_preprocess import read_data_from_csv
    elif dataset_name in assist2015_list:
        from .assist2015_preprocess import read_data_from_csv
    elif dataset_name == "algebra2005":
        from .algebra2005_preprocess import read_data_from_csv
    elif dataset_name == "bridge2algebra2006":
        from .bridge2algebra2006_preprocess import read_data_from_csv
    elif dataset_name == "statics2011":
        from .statics2011_preprocess import read_data_from_csv
    elif dataset_name == "nips_task34":
        from .nips_task34_preprocess import read_data_from_csv
    elif dataset_name == "poj":
        from .poj_preprocess import read_data_from_csv
    elif dataset_name == "ednet":
        from .ednet_preprocess import read_data_from_csv

    if dataset_name != "nips_task34":
        read_data_from_csv(readf, writef)
    else:
        metap = os.path.join(dname, "metadata")
        read_data_from_csv(readf, metap, "task_3_4", writef)
     
    return dname,writef