import numpy as np
from IPython import embed
import pickle 

def sta_infos(df, keys, stares, split_str="_"):
    # keys: 0: uid , 1: concept, 2: question
    uids = df[keys[0]].unique()
    if len(keys) == 2:
        cids = df[keys[1]].unique()
    elif len(keys) > 2:
        qids = df[keys[2]].unique()
        ctotal = 0
        cq = df.drop_duplicates([keys[2], keys[1]])[[keys[2], keys[1]]]
        cq[keys[1]] = cq[keys[1]].fillna("NANA")
        cids, dq2c = set(), dict()
        for i, row in cq.iterrows():
            q = row[keys[2]]
            ks = row[keys[1]]
            dq2c.setdefault(q, set())
            if ks == "NANA":
                continue
            for k in ks.split(split_str):
                dq2c[q].add(k)
                cids.add(k)
        ctotal, na, qtotal = 0, 0, 0
        for q in dq2c:
            if len(dq2c[q]) == 0:
                na += 1 # questions has no concept
                continue
            qtotal += 1
            ctotal += len(dq2c[q])
        
        avgcq = round(ctotal / qtotal, 4)
    avgins = round(df.shape[0] / len(uids), 4)
    ins, us, qs, cs = df.shape[0], len(uids), "NA", len(cids)
    avgcqf, naf = "NA", "NA"
    if len(keys) > 2:
        qs, avgcqf, naf = len(qids), avgcq, na
    curr = [ins, us, qs, cs, avgins, avgcqf, naf]
    stares.append(",".join([str(s) for s in curr]))
    return ins, us, qs, cs, avgins, avgcqf, naf

def ednet_sta_infos(df, content_df, keys, split_str="_"):
    # keys: 0: user_id , 1: question_id, 2: user_answer, 3: correct_answer
    try:
        uids = df[keys[0]].unique()
        cids = df[keys[1]].unique()
        correct = df[keys[2]].values == content_df[content_df[keys[1]].isin(df[keys[1]])].loc[:,[keys[3]]].values.flatten()
        correct = np.array(correct).astype(int)
        df['correct'] = correct
        
        ins, us, qs, cs = df.shape[0], len(uids), 0, len(cids)
        return np.array([ins, us, qs, cs])
    except:
        return np.array([0, 0, 0, 0])

def write_txt(file, data):
    with open(file, "w") as f:
        for dd in data:
            for d in dd:
                f.write(",".join(d) + "\n")

def save_pickle(file, data):
    with open(file, 'wb') as f: 
        pickle.dump(data, f)
        print(f"saved at {file}")

from datetime import datetime
def change2timestamp(t, hasf=True):
    if hasf:
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000
    else:
        timeStamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
    return int(timeStamp)

def replace_text(text):
    text = text.replace("_", "####").replace(",", "@@@@")
    return text