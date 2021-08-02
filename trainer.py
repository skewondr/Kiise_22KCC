import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from itertools import repeat, chain, islice
import os
from logzero import logger

from config import ARGS
from network.util_network import ScheduledOptim, NoamOpt

class EarlyStopping:
    #https://stats.stackexchange.com/questions/68893/area-under-curve-of-roc-vs-overall-accuracy
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_auc, model):

        score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, os.path.join(self.path, 'best_ckpt.pt'))
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.save_checkpoint(score, model, os.path.join(self.path, 'last_ckpt.pt'))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model, os.path.join(self.path, 'best_ckpt.pt'))
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model, path):
        if self.verbose:
            logger.info(f'[Validation] AUC ({self.best_score:.6f} --> {score:.6f}). Saving model {path}')
        torch.save(model.state_dict(), path)

class NoamOptimizer:
    def __init__(self, model, lr, model_size, warmup):
        self._adam = torch.optim.Adam(model.parameters(), lr=lr)
        self._opt = NoamOpt(
            model_size=model_size, factor=1, warmup=warmup, optimizer=self._adam)

    def step(self, loss):
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()


class Trainer:
    def __init__(self, model, device, warm_up_step_count, eval_steps,
                 d_model, num_epochs, weight_path, lr, es_patience,
                 train_data, val_data, test_data):
        self._device = device
        self._num_epochs = num_epochs
        self._weight_path = weight_path

        self._model = model
        self._loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self._model.to(device)

        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data

        self._opt = NoamOptimizer(model=model, lr=lr, model_size=d_model, warmup=warm_up_step_count)

        self.global_step = 0
        self._threshold = 0.5
        self.max_acc = 0.0
        self.max_auc = 0.0

        self.test_acc = 0.0
        self.test_auc = 0.0
        self.es_patience = es_patience
        self.eval_steps = eval_steps
        self.early_stopping = EarlyStopping(patience=self.es_patience, verbose=True, path=self._weight_path)   

    # train model and choose weight with max auc on validation dataset
    def train(self):
        train_gen = data.DataLoader(
            dataset=self._train_data,
            shuffle=True, 
            pin_memory=False if ARGS.device == 'cpu' else True,
            batch_size=ARGS.train_batch, 
            num_workers=ARGS.num_workers, 
            collate_fn=self._train_data.get_sequence
        )
        val_gen = data.DataLoader(
            dataset=self._val_data, 
            shuffle=False, 
            pin_memory=False if ARGS.device == 'cpu' else True,
            batch_size=ARGS.test_batch, 
            num_workers=ARGS.num_workers, 
            collate_fn=self._val_data.get_sequence
        )

        self._model.train()
        for epoch in range(self._num_epochs):
            start_time = time.time()
            
            losses = []
            num_corrects = 0
            num_total = 0
            labels = []
            outs = []

            for batch_idx, batch in tqdm(enumerate(train_gen), total=len(train_gen), ncols=100):
                label, out, pred = self._forward(batch)
                train_loss = self._get_loss(label, out)
                losses.append(train_loss.item())
                if batch_idx % 100 == 0:    
                    logger.info(f'{epoch} {batch_idx * ARGS.train_batch}/{len(train_gen) * ARGS.train_batch} early stop: {self.early_stopping.counter}/{self.es_patience}, loss: {train_loss:.4f}')

                self._opt.step(train_loss)

                num_corrects += (pred == label).sum().item()
                num_total += len(label)

                labels.extend(label.squeeze(-1).data.cpu().numpy())
                outs.extend(out.squeeze(-1).data.cpu().numpy())

                if batch_idx * ARGS.train_batch % self.eval_steps == 0 and batch_idx != 0:
                    self._test('Validation', val_gen, epoch)
                    
                if self.early_stopping.early_stop: break

            acc = num_corrects / num_total
            try: auc = roc_auc_score(labels, outs)
            except ValueError: auc = 0.0
            
            loss = np.mean(losses)
            training_time = time.time() - start_time

            logger.info(f'{epoch} (total) early stop: {self.early_stopping.counter}/{self.es_patience}, loss: {loss:.4f}, acc: {acc:.4f}, auc: {auc:.4f}, time: {training_time:.2f}')
                
            if self.early_stopping.early_stop: 
                logger.info("Early stopped...")
                break
            

    # get test results
    def test(self):
        test_gen = data.DataLoader(
            dataset=self._test_data, 
            shuffle=False, 
            pin_memory=False if ARGS.device == 'cpu' else True,
            batch_size=ARGS.test_batch, 
            num_workers=ARGS.num_workers, 
            collate_fn=self._test_data.get_sequence
        )
        # load best weight
        self._model.load_state_dict(torch.load(os.path.join(self._weight_path, 'best_ckpt.pt')))
        logger.info('-----------------------------------------------------------------------------------------------------------')
        self._test('Test', test_gen, 0)
        logger.info('-----------------------------------------------------------------------------------------------------------')

    def _forward(self, batch):
        batch = {k: t.to(self._device) for k, t in batch.items()}
        label = batch['label']  # shape: (batch_size, 1)

        output = self._model(batch['input'], batch['target_id'])
        pred = (torch.sigmoid(output) >= self._threshold).long()  # shape: (batch_size, 1)

        return label, output, pred

    def _get_loss(self, label, output):
        loss = self._loss_fn(output, label.float())
        return loss.mean()

    # takes iterable
    def _test(self, name, batches, epoch):
        start_time = time.time()
        self._model.eval()

        losses = []
        num_corrects = 0
        num_total = 0
        labels = []
        outs = []

        with torch.no_grad():
            for batch in tqdm(batches, total=len(batches), ncols=100):
                label, out, pred = self._forward(batch)
                test_loss = self._get_loss(label, out)
                losses.append(test_loss.item())

                num_corrects += (pred == label).sum().item()
                num_total += len(label)

                labels.extend(label.squeeze(-1).data.cpu().numpy())
                outs.extend(out.squeeze(-1).data.cpu().numpy())

        acc = num_corrects / num_total
        auc = roc_auc_score(labels, outs)
        loss = np.mean(losses)
        training_time = time.time() - start_time
        
        if name == 'Validation':
            self.early_stopping(auc, self._model)
        logger.info(f'[{name}] (total) early stop: {self.early_stopping.counter}/{self.es_patience}, loss: {loss:.4f}, acc: {acc:.4f}, auc: {auc:.4f}, time: {training_time:.2f}')
        

