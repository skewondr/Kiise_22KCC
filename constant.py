from config import ARGS
import os
from logzero import logger
import pickle 

MIN_LENGTH = 2
MAX_LENGTH = 100
PAD_INDEX = 0
MASK_PAD_INDEX = -100

QUESTION_NUM = {
    'modified_AAAI20': 16175,
    'ASSISTments2009': 110,
    'ASSISTments2012': 0, # TODO: fix
    'ASSISTments2015': 100,
    'ASSISTmentsChall': 102,
    'KDDCup': 0, # TODO: fix
    'Junyi': 722,
    'STATICS': 1223,
    'EdNet-KT1': 18143
}

acc_name = f"../dataset/{ARGS.dataset_name}/processed/1/sub{ARGS.sub_size}/train_{ARGS.sub_size}acc.pickle"
if os.path.isfile(acc_name):
    with open(acc_name, 'rb') as f: 
        ACC_DICT = pickle.load(f)
        logger.info("open accuracy dict")