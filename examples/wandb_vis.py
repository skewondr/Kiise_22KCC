import matplotlib.pyplot as plt
import pickle 
import seaborn as sns
from collections import Counter
import numpy as np
import os
from IPython import embed 

def load_p (name):
  with open(name, 'rb') as f:
    cnt = sorted(Counter(pickle.load(f)).items())
    # print(cnt)
    x = [i[0] for i in cnt]
    y = [i[1] for i in cnt]
    return np.array(x), np.array(y)

# xa_cat11_o, ya_cat11_o = load_p("lcrts_o.pickle")
# xa_cat11_x, ya_cat11_x = load_p("lcrts_x.pickle")

def viz(save_dict):
  for key, save_dir in save_dict.items():
    dataset_name = save_dir.split("_")[0]
    model_name = save_dir.split("_")[1]

    xa_ori11_o, ya_ori11_o = load_p(os.path.join("saved_model/"+save_dir, "lcrts_o.pickle"))
    xa_ori11_x, ya_ori11_x = load_p(os.path.join("saved_model/"+save_dir, "lcrts_x.pickle"))
    sns.set_palette("Set2", 4)
    figsize = (5, 4)
    plt.figure(figsize=figsize)
    plt.barh(xa_ori11_x[5:], ya_ori11_x[5:]*(-1), label="FP")
    plt.barh(xa_ori11_x[:5]-0.5, ya_ori11_x[:5]*(-1), label="FN")
    plt.barh(xa_ori11_o[5:], ya_ori11_o[5:], label="TN")
    plt.barh(xa_ori11_o[:5]-0.5, ya_ori11_o[:5], label="TP")
    plt.axvline(0.0, 0.0, 1.0, color='white', linestyle='--', linewidth=2)
    plt.axhline(4.25, 0.0, 1.0, color='lightgray', linestyle='--', linewidth=2)

    # plt.gca().invert_xaxis()
    plt.legend(loc='best', ncol=2)
    # plt.xticks(np.arange(-60000, 60000, 10000), labels=['100000', '50000', '0', '50000', '100000', '150000', '200000'])
    plt.xticks(np.arange(-1*max(ya_ori11_x), max(ya_ori11_o), int(max(ya_ori11_o)/3)))
    plt.tick_params(left=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    # plt.gca().axes.yaxis.set_visible(False)
    plt.title(f'Prediction of {model_name} ({key})')
    plt.xlabel(f'{dataset_name} Test Dataset')
    plt.ylabel('Correct Answer Rate', labelpad=30)
    plt.savefig(f'{key}_{save_dir.split("_")[:2]}.png')
    # plt.show()

def viz2(save_dict):
  for key, save_dir in save_dict.items():
    dataset_name = save_dir.split("_")[0]
    model_name = save_dir.split("_")[1]

    xa_ori11_o, ya_ori11_o = load_p(os.path.join("saved_model/"+save_dir, "lcrts_o.pickle"))
    xa_ori11_x, ya_ori11_x = load_p(os.path.join("saved_model/"+save_dir, "lcrts_x.pickle"))
    ylbo =  ya_ori11_o[:5] + ya_ori11_x[:5]
    ylbx = ya_ori11_o[5:] + ya_ori11_x[5:]
    print(f"label 0 : {ylbx}, {np.sum(ylbx)}, label 1: {ylbo}, {np.sum(ylbo)} = {np.sum(ylbx)+ np.sum(ylbo)}")
    print("\n")
    sns.set_palette("Accent", 2)
    figsize = (10, 6)
    plt.figure(figsize=figsize)
    plt.barh(xa_ori11_x[5:], ylbx, label="label 0")
    plt.barh(xa_ori11_o[:5]-0.5, ylbo, label="label 1")
    # plt.axvline(0.0, 0.0, 1.0, color='white', linestyle='--', linewidth=2)
    plt.axhline(4.25, 0.0, 1.0, color='lightgray', linestyle='--', linewidth=2)

    # plt.gca().invert_xaxis()
    plt.legend(loc='best', ncol=2)
    # plt.xticks(np.arange(-60000, 60000, 10000), labels=['100000', '50000', '0', '50000', '100000', '150000', '200000'])
    plt.xticks(np.arange(0, max(max(ylbx), max(ylbo)), int(max(ylbo)/3)))
    plt.tick_params(left=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    # plt.gca().axes.yaxis.set_visible(False)
    plt.title(f'Distribution of Classes')
    plt.xlabel(f'{dataset_name} Test Dataset')
    plt.ylabel('Correct Answer Rate', labelpad=30)
    plt.savefig(f'{key}_{save_dir.split("_")[1:]}.png')

def viz3(save_dict):
    cnt = 0 
    for key, save_dir in save_dict.items():
        try:
            dataset_name = save_dir.split("_")[0]
            model_name = save_dir.split("_")[1]

            xa_ori11_o, ya_ori11_o = load_p(os.path.join("saved_model/"+save_dir, "lcrts_o.pickle"))
            xa_ori11_x, ya_ori11_x = load_p(os.path.join("saved_model/"+save_dir, "lcrts_x.pickle"))
            ylbo =  ya_ori11_o[:5] + ya_ori11_x[:5]
            ylbx = ya_ori11_o[5:] + ya_ori11_x[5:]
            print(save_dir)
            print(f"label 0 : {ylbx}, {np.sum(ylbx)}, label 1: {ylbo}, {np.sum(ylbo)} = {np.sum(ylbx)+ np.sum(ylbo)}")
            # print(f"same? {np.sum(ya_ori11_x) + np.sum(ya_ori11_o)}")
            print()
            sns.set_palette("Accent", 6)
            figsize = (6, 8)
            if cnt == 0: 
                plt.figure(figsize=figsize)
                plt.barh(xa_ori11_x[5:], ylbx, label="label 0")
                plt.barh(xa_ori11_o[:5]-0.5, ylbo, label="label 1")
                cnt += 1
            plt.barh(xa_ori11_x[5:], ya_ori11_o[5:], label=f"TN_{key}")
            plt.barh(xa_ori11_o[:5]-0.5, ya_ori11_o[:5], label=f"TP_{key}")
            # plt.axvline(0.0, 0.0, 1.0, color='white', linestyle='--', linewidth=2)
            plt.axhline(4.25, 0.0, 1.0, color='lightgray', linestyle='--', linewidth=2)

            # plt.gca().invert_xaxis()
            plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)
            # plt.legend(loc='best', ncol=2)
            # plt.xticks(np.arange(-60000, 60000, 10000), labels=['100000', '50000', '0', '50000', '100000', '150000', '200000'])
            plt.xticks(np.arange(0, max(max(ylbx), max(ylbo))+1, int(max(ylbo)/3)))
            plt.tick_params(left=False)
            plt.setp(plt.gca().get_yticklabels(), visible=False)
            # plt.gca().axes.yaxis.set_visible(False)
            plt.title(f'Distribution of Classes', y=1.1)
            plt.xlabel(f'{dataset_name} Test Dataset')
            plt.ylabel('Correct Answer Rate', labelpad=30)
            # plt.savefig(f'predict_{key}_{save_dir.split("_")[1:]}.png')
        except:
            print(f"{save_dir} not in server 7")


ednet = {
# "1": "ednet_saint_42_5_R_sinu_5_828165147",#7 x
"2": "ednet_sakt_42_5_R_sinu_5_82611443",#8 ok
# "3": "ednet_dkvmn_42_5_R_sinu_5_828175429",#7 ok
# "4": "ednet_dkt_42_5_R_sinu_5_826115140",#7 ok re

# "5": "ednet_dkvmn_42_5_R_add_5_828101915",#7 ok
# "6": "ednet_sakt_42_5_R_add_5_82764238",#7 ok
# "7": "ednet_saint_42_5_R_add_5_82891914",#7 x
# "8": "ednet_dkt_42_5_R_add_5_82755354",#7 ok re

"9": "ednet_sakt_42_5_R_quantized_5_826105935",#8 ok
}


as09 = {
"1": "assist2009_dkt_42_5_R_sinu_5_827173527",#8 ok 
# "2": "assist2009_dkvmn_42_5_R_sinu_5_828205856",#7 ok 
# "3": "assist2009_saint_42_5_R_sinu_5_828192619",#7 x
"4": "assist2009_sakt_42_5_R_sinu_5_82717330",#8 ok 

# "5": "assist2009_dkvmn_42_5_R_add_5_828121558",#7 ok 
# "6": "assist2009_saint_42_5_R_add_5_828105358",#7 x
"7": "assist2009_sakt_42_5_R_add_5_82773336",#8 ok 
"8": "assist2009_dkt_42_5_R_add_5_8277017",#8 ok 

"9": "assist2009_sakt_42_5_R_quantized_5_827143449",#8 ok 
}

# save_dict = {
# "base": "ednet_sakt_3407_5_qid_82961526",
# "sinu@5": "ednet_sakt_224_5_R_sinu_5_826111535",
# }

# save_dict = {
# "base": "ednet_saint_3407_5_qid_8297044",
# }

# save_dict = {
# # "qid": "assist2009_dkt_42_5_16223023",
# "pre-trained": "assist2009_dkt_42_5_1732635",
# # "qid_0": "assist2009_dkvmn_42_5_1623286"
# # "qid": "ednet_dkt_42_5_20171956",
# # "pre-trained": "ednet_dkt_42_5_2018366",
# # "qid_0": "ednet_dkt_42_5_2019231"
# }
viz3(as09)
