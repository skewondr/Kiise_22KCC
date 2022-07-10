import pandas as pd
from .utils import ednet_sta_infos, write_txt, change2timestamp, save_pickle
import os
import re
from tqdm import tqdm 
import numpy as np
from random import sample
from IPython import embed
import sys
import pickle 
import statistics
from collections import Counter

"""
* ins: data 개수 df.shape[0]
* us: 유저 개수 len(uids)
* cs: KC 개수 len(cids). KC가 없는 데이터의 경우, Question ID가 들어감
* avgins: 유저 한명 당 평균적인 기록 개수 
<문제, KC 둘 다 있는 경우 >
* qs: Question ID
* avgcqf: 문제 하나 당 평균적인 KC 개수 
* naf: KC가 없는 Question의 개수  
"""
content_path =  "/home/tako/yoonjin/pykt-toolkit/data/EdNet/EdNet-Contents/contents/questions.csv"
content_df = pd.read_csv(content_path)

KEYS = ["user_id", "question_id", "user_answer", "correct_answer"]

def read_data_from_csv(read_file, write_file):
    stares = []
    user_inters = []
    sample_stat = np.array([0, 0, 0, 0]) #ins, us, qs, cs
    af_sample_stat = np.array([0, 0, 0, 0]) #ins, us, qs, cs

    total_q_1 = Counter({})
    total_q_n = Counter({})
    total_q_acc = {}

    user_path_list = os.listdir(read_file)
    user_path_list = sample(user_path_list, int(len(user_path_list)*(0.01)))
    us = len(user_path_list)

    dname = "/".join(write_file.split("/")[0:-1])
    save_name = os.path.join(dname, "correct_rate.pickle")

    for idx, user_path in enumerate(tqdm(user_path_list, total=us, ncols=100)):
        uid = user_path.split('/')[-1]
        uid = int(re.sub(r'[^0-9]', '', uid))

        df = pd.read_csv(os.path.join(read_file, user_path), encoding = 'ISO-8859-1', dtype=str)
        df["user_id"] = uid

        tmp = ednet_sta_infos(df, content_df, KEYS)
        sample_stat += tmp

        _df = df.dropna(subset=["user_id", "question_id", "correct", "timestamp"])
        tmp = ednet_sta_infos(df, content_df, KEYS)
        af_sample_stat += tmp

        tmp_inter = _df
        tmp_inter = tmp_inter.sort_values(by=['timestamp'])
        seq_len = len(tmp_inter)
        tmp_inter["question_id"] = tmp_inter["question_id"].str.replace(pat=r'q', repl=r'', regex=True)
        seq_skills = tmp_inter['question_id'].astype(str)
        seq_ans = tmp_inter['correct'].astype(str)
        seq_problems = ["NA"]
        seq_start_time = ["NA"]
        seq_response_cost = ["NA"]

        assert seq_len == len(seq_skills) == len(seq_ans)

        user_inters.append(
            [[str(uid), str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_response_cost])

        q_cnt_df = tmp_inter.groupby(['question_id', 'correct']).size().unstack(fill_value=0)
        # print(q_cnt_df)
        if len(q_cnt_df.columns) > 1:
            q_cnt_df['num'] = q_cnt_df.iloc[:, 0] + q_cnt_df.iloc[:, 1]
            q_1 = q_cnt_df.iloc[:, 1].to_dict()
            q_n = q_cnt_df['num'].to_dict()
        else: 
            if q_cnt_df.columns[0] == 1:
                q_1 = q_cnt_df.iloc[:, 0].to_dict()
            else:
                q_1 = {} 
            q_n = q_cnt_df.iloc[:, 0].to_dict()
        
        # q_1 update
        total_q_1 = total_q_1+ Counter(q_1)
        # q_n update 
        total_q_n = total_q_n+ Counter(q_n)

    ins, us, qs, cs = sample_stat[0], sample_stat[1], sample_stat[2], sample_stat[3]
    avgins = round(ins / us, 4)
    # avgcq = round(cs / qs, 4)
    avgcq, na = "NA", "NA"
    curr = [ins, us, qs, cs, avgins, avgcq, na]
    stares.append(",".join([str(s) for s in curr]))
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    ins, us, qs, cs= af_sample_stat[0], af_sample_stat[1], af_sample_stat[2], af_sample_stat[3]
    avgins = round(ins / us, 4)
    # avgcq = round(cs / qs, 4)
    avgcq, na = "NA", "NA"
    curr = [ins, us, qs, cs, avgins, avgcq, na]
    stares.append(",".join([str(s) for s in curr]))
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    # save correct ratio
    for key in total_q_1.keys():
        if total_q_n[key] > 0:
            total_q_acc[key] = total_q_1[key]/total_q_n[key]

    print(f"average of question accuracy:{statistics.mean(list(total_q_acc.values())):.2f}")
    save_pickle(save_name, total_q_acc)

    return

