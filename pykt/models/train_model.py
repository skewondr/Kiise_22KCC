import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
from .atkt import _l2_normalize_adv
from ..utils.utils import debug_print
from IPython import embed
from .emb import EMB

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cal_loss(model, ys, r, rshft, sm, preloss=[]):
    model_name = model.model_name

    if model_name in ["dkt", "dkt_forget", "dkvmn", "kqn", "sakt", "saint", "atkt", "atktfix", "gkt"]:

        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y, t)
    elif model_name == "dkt+":
        y_curr = torch.masked_select(ys[1], sm)
        y_next = torch.masked_select(ys[0], sm)
        r_curr = torch.masked_select(r, sm)
        r_next = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y_next, r_next)

        loss_r = binary_cross_entropy(y_curr, r_curr) # if answered wrong for C in t-1, cur answer for C should be wrong too
        loss_w1 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=1, dim=-1), sm[:, 1:])
        loss_w1 = loss_w1.mean() / model.num_c
        loss_w2 = torch.masked_select(torch.norm(ys[2][:, 1:] - ys[2][:, :-1], p=2, dim=-1) ** 2, sm[:, 1:])
        loss_w2 = loss_w2.mean() / model.num_c

        loss = loss + model.lambda_r * loss_r + model.lambda_w1 * loss_w1 + model.lambda_w2 * loss_w2
    elif model_name == "akt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y, t) + preloss[0]

    return loss


def model_forward(device, model, dataset_name, data):
    model_name = model.model_name
    emb_type = model.emb_type
    if model_name in ["dkt_forget"]:
        q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
    elif model_name in ["saint", "akt"]:
        q, c, r, qshft, cshft, rshft, m, sm, q_diff, c_diff = data
    elif emb_type != "qid" or dataset_name in ["assist2015", "ednet"]:
        q, c, r, qshft, cshft, rshft, m, sm, q_diff, c_diff = data
    else: 
        c, q, r, cshft, qshft, rshft, m, sm, c_diff, q_diff = data

    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)
    mm = torch.cat([torch.ones((m.shape[0], 1), dtype=torch.bool).to(device), m], dim=1)

    if model_name in ["dkt"]:
        y = model(c.long(), r.long(), c_diff[:,:-1].long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y) # first: yshft
    elif model_name == "dkt+":
        y = model(c.long(), r.long())
        y_next = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        y_curr = (y * one_hot(c.long(), model.num_c)).sum(-1)
        ys = [y_next, y_curr, y]
    elif model_name in ["dkt_forget"]:
        y = model(c.long(), r.long(), d, dshft)
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        ys.append(y)
    elif model_name in ["dkvmn"]:
        y = model(cc.long(), cr.long())
        ys.append(y[:,1:])
    elif model_name in ["kqn", "sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])
    elif model_name == "akt":               
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(reg_loss)
    elif model_name in ["atkt", "atktfix"]:
        y, features = model(c.long(), r.long())
        y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
        loss = cal_loss(model, [y], r, rshft, sm)
        # at
        features_grad = grad(loss, features, retain_graph=True)
        p_adv = torch.FloatTensor(model.epsilon * _l2_normalize_adv(features_grad[0].data))
        p_adv = Variable(p_adv).to(device)
        pred_res, _ = model(c.long(), r.long(), p_adv)
        # second loss
        pred_res = (pred_res * one_hot(cshft.long(), model.num_c)).sum(-1)
        adv_loss = cal_loss(model, [pred_res], r, rshft, sm)
        loss = loss + model.beta * adv_loss
    elif model_name == "gkt":
        y = model(cc.long(), cr.long())
        ys.append(y)  
    elif model_name == "emb":
        mse_loss = nn.MSELoss()
        y = model(cc.long())
        loss = mse_loss(torch.masked_select(y, mm), torch.masked_select(c_diff, mm))
    # cal loss
    if model_name not in ["atkt", "atktfix", "emb"] :
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
    if emb_type.startswith("qid_"):
        emb_model = EMB(model.num_c, 512, 0.5, emb_type="qid").to(device) 
        mse_loss = nn.MSELoss()
        y = emb_model(cc.long())
        loss2 = mse_loss(torch.masked_select(y, mm), torch.masked_select(c_diff, mm))
        lambda_ = float(emb_type.split("_")[-1])
        assert lambda_ >= 0, "set proper lambda"
        loss = loss + lambda_*loss2 
        embed()
    return loss
    

def train_model(device, fold, model, dataset_name, train_loader, valid_loader, num_epochs, opt, ckpt_path, early_stopping, test_loader=None, test_window_loader=None, save_model=False):
    max_auc, best_epoch, min_loss = 0, -1, 100
    train_step = 0
    for i in range(1, num_epochs + 1):
        loss_mean = []
        for data in train_loader:
            train_step+=1
            model.train()
            loss = model_forward(device, model, dataset_name, data)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_mean.append(loss.detach().cpu().numpy())
            if model.model_name == "gkt" and train_step%10==0:
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text = text,fuc_name="train_model")


        loss_mean = np.mean(loss_mean)
        auc, acc, mse = evaluate(device, model, dataset_name, valid_loader, model.model_name)
        ### atkt 有diff， 以下代码导致的
        ### auc, acc = round(auc, 4), round(acc, 4)

        if model.model_name == "emb":
            if mse < min_loss:
                if save_model:
                    torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type+f"_model_{fold}.ckpt"))
                min_loss = mse
                best_epoch = i
                testauc, testacc = -1, -1
                window_testauc, window_testacc = -1, -1
                if not save_model:
                    if test_loader != None:
                        save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                        testauc, testacc, test_mse = evaluate(device, model, dataset_name, test_loader, model.model_name, save_test_path)
                    if test_window_loader != None:
                        save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                        window_testauc, window_testacc, window_testmse= evaluate(device, model, dataset_name, test_window_loader, model.model_name, save_test_path)
                    testauc, testacc, window_testauc, window_testacc = round(testauc, 4), round(testacc, 4), round(window_testauc, 4), round(window_testacc, 4)
            # window_testauc, window_testacc = -1, -1
            validauc, validacc, validmse = round(auc, 4), round(acc, 4), round(mse, 4)#model.evaluate(valid_loader, emb_type)
            # trainauc, trainacc = model.evaluate(train_loader, emb_type)
            # max_auc = round(max_auc, 4)
            print(f"Epoch: {i}, validmse: {validmse:.4f}, best epoch: {best_epoch:.4f}, best min_loss: {min_loss:.4f}, train loss: {loss_mean:.4f}")
            # print(f"            testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")
    
            if i - best_epoch > early_stopping.patience: 
                print("Early stopped...")
                break

        else: 
            if auc > max_auc:
                if save_model:
                    torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type+f"_model_{fold}.ckpt"))
                max_auc = auc
                best_epoch = i
                testauc, testacc = -1, -1
                window_testauc, window_testacc = -1, -1
                if not save_model:
                    if test_loader != None:
                        save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                        testauc, testacc, _ = evaluate(device, model, dataset_name, test_loader, model.model_name, save_test_path)
                    if test_window_loader != None:
                        save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                        window_testauc, window_testacc, _ = evaluate(device, model, dataset_name, test_window_loader, model.model_name, save_test_path)
                    testauc, testacc, window_testauc, window_testacc = round(testauc, 4), round(testacc, 4), round(window_testauc, 4), round(window_testacc, 4)
            # window_testauc, window_testacc = -1, -1
            validauc, validacc, validmse = round(auc, 4), round(acc, 4), round(mse, 4)#model.evaluate(valid_loader, emb_type)
            # trainauc, trainacc = model.evaluate(train_loader, emb_type)
            max_auc = round(max_auc, 4)
            print(f"Epoch: {i}, validauc: {validauc:.4f}, validacc: {validacc:.4f}, best epoch: {best_epoch:.4f}, best auc: {max_auc:.4f}, loss: {loss_mean:.4f}")
            # print(f"            testauc: {testauc}, testacc: {testacc}, window_testauc: {window_testauc}, window_testacc: {window_testacc}")
    
            early_stopping(auc, i)

            if early_stopping.early_stop:
                print("Early stopped...")
                break

    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, validmse, best_epoch