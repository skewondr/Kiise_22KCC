import os, sys
from .split_datasets import main as split
import shutil

def process_raw_data(dataset_name, kfold, subset, dname2paths):
    readf = dname2paths[dataset_name]
    if dataset_name == "ednet": 
        dname = "/".join(readf.split("/")[0:-2]) + f"/sub{subset}_fold{kfold}"
    else: 
        dname = "/".join(readf.split("/")[0:-1]) + f"/sub{subset}_fold{kfold}"

    if os.path.exists(dname):
        shutil.rmtree(dname)
    os.makedirs(dname)

    writef = os.path.join(dname, "data.txt")
    print(f"Start preprocessing data: {dataset_name}")
    if dataset_name == "assist2009":
        from .assist2009_preprocess import read_data_from_csv
    elif dataset_name in "assist2015":
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

    if dataset_name == "ednet":
        read_data_from_csv(readf, writef, subset)
    elif dataset_name != "nips_task34":
        read_data_from_csv(readf, writef)
    else:
        metap = os.path.join(dname, "metadata")
        read_data_from_csv(readf, metap, "task_3_4", writef)
     
    return dname,writef