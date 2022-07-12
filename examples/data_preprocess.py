import os, sys
import argparse
from pykt.preprocess import process_raw_data
from pykt.preprocess.split_datasets import main as split
from data_config import ARGS as args

dname2paths = {
    # "assist2009": "../data/assist2009/skill_builder_data_corrected_collapsed.csv",
    "assist2009": "../data/assist2009/skill_builder_data_corrected.csv",
    "assist2015": "../data/assist2015/2015_100_skill_builders_main_problems.csv",
    "algebra2005": "../data/algebra2005/algebra_2005_2006_train.txt",
    "bridge2algebra2006": "../data/bridge2algebra2006/bridge_to_algebra_2006_2007_train.txt",
    "statics2011": "../data/statics2011/AllData_student_step_2011F.csv",
    "nips_task34": "../data/nips_task34/train_task_3_4.csv",
    "poj": "../data/poj/poj_log.csv",
    "ednet": "../data/EdNet/EdNet-KT1/KT1/",
}
configf = "../configs/data_config.json"

if __name__ == "__main__":
    # process raw data
    dname, writef = process_raw_data(args.dataset_name, args.kfold, args.subset, dname2paths)
    print("-"*50)
    # split
    os.system("rm " + dname + "/*.pkl")
    split(dname, writef, args.dataset_name, configf, args.min_seq_len, args.maxlen, args.kfold)
    print("="*100)
