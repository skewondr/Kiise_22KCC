from nbformat import write
import pandas as pd
from .utils import sta_infos, write_txt, save_pickle
from collections import Counter
import statistics
import os
from IPython import embed
import numpy as np
import re

KEYS = ["user_id", "sequence_id"]

def seq2acc(write_file, seq_df, q_acc):
    seq_skills = seq_df['sequence_id'].astype(str)

    k = int(re.sub("[^0-9]", "", write_file.split("/")[-2].split("_")[-2]))
    quantile = np.quantile(list(q_acc.values()), q=np.arange(0,1,1/k))
    
    acc = []
    for i, v in seq_skills.items():
        if int(v) in q_acc.keys():
            score = q_acc[int(v)]
        else: 
            score = 0
        idx = np.abs(quantile - score).argmin()
        if quantile.flat[idx] <= score:
            idx +=1
        acc.append(idx)
    seq_df['acc'] = acc
    return seq_df['acc'].astype(str)


def read_data_from_csv(read_file, write_file):
    stares = []

    df = pd.read_csv(read_file)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = range(df.shape[0])

    df = df.dropna(subset=["user_id", "log_id", "sequence_id", "correct"])
    df = df[df['correct'].isin([0,1])]#filter responses
    df['correct'] = df['correct'].astype(int)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    print("\n".join(stares))

    dname = "/".join(write_file.split("/")[0:-1])
    save_name = os.path.join(dname, "correct_rate.pickle")
    
    q_cnt_df = df.groupby(['sequence_id', 'correct']).size().unstack(fill_value=0)
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

    total_q_1 = Counter(q_1)
    total_q_n = Counter(q_n)
    
    # save correct ratio
    total_q_acc = {}
    for key in total_q_1.keys():
        if total_q_n[key] > 0:
            total_q_acc[key] = total_q_1[key]/total_q_n[key]

    print(f"average of question accuracy:{statistics.mean(list(total_q_acc.values())):.2f}")
    save_pickle(save_name, total_q_acc)

    ui_df = df.groupby(['user_id'], sort=False)

    user_inters = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=["log_id", "index"])
        seq_len = len(tmp_inter)
        if "q2a" in write_file.split("/")[-2] :
            seq_skills = seq2acc(write_file, tmp_inter, total_q_acc)
        else: 
            seq_skills = tmp_inter['sequence_id'].astype(str)
        seq_ans = tmp_inter['correct'].astype(str)
        seq_problems = ["NA"]
        seq_start_time = ["NA"]
        seq_response_cost = ["NA"]

        assert seq_len == len(seq_skills) == len(seq_ans)

        user_inters.append(
            [[str(user), str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_response_cost])

    write_txt(write_file, user_inters)

    return

