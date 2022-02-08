import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import argparse
import random
from model.config import *
from torch.utils.data import DataLoader
from lib.dataloader import MyDataset,collate_fn
from lib.metrics import *
import os
from lib.logger import get_logger
import copy
import time
from tqdm import tqdm


class trainer():
    def __init__(self,args,config,model,adj_mx):
        super(trainer, self).__init__()
        self.config = config
        self.model = model
        self.adj_mx = adj_mx
       # self.data = data
        self.lr = config.base_lr
        self.weight_decay = config.weight_decay
        self.eps = config.epsilon
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,eps=self.eps)
        self.scaeduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=10, threshold=1e-3,
                                                              min_lr=1e-5, verbose=True)
        self.loss = self.get_loss()
        self.epoch_num  = config.epochs
        self.args = args
        self.best_path = os.path.join(self.config.log_dir, 'best_model.pth')

        #load  dataloader
        self.train_set = MyDataset(config,train=True)
        self.test_set = MyDataset(config,train=False,test=True)
        self.valid_set= MyDataset(config,train=False,test=False,valid=True)
        self.train_loader = DataLoader(self.train_set,batch_size=config.batch_size,collate_fn=collate_fn,shuffle=True, num_workers=32,drop_last=True)
        self.test_loader = DataLoader(self.test_set,batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn,num_workers=32,drop_last=True)
        self.valid_loader = DataLoader(self.valid_set,batch_size=config.batch_size,collate_fn=collate_fn,shuffle=True,num_workers=32,drop_last=True)
        self.train_per_epoch = len(self.train_loader)
        self.scaler = self.train_set.scaler


        #log
        if os.path.isdir(self.config.log_dir) == False and args.is_train:
            os.makedirs(self.config.log_dir,exist_ok=True)
        self.logger = get_logger(self.config.log_dir,name=self.config.model,debug=not args.is_train)
        self.logger.info('Experiment log path in {}'.format(config.log_dir))
        if  args.is_train:
            self.logger.info("Aegument:%r",args)

    def get_loss(self):
        if self.config.loss_func == 'mae':
            #loss = torch.nn.L1Loss()
            loss = MAE_mask
        elif self.config.loss_func == 'rmse':
            loss = RMSE_mask
        elif self.config.loss_func == 'mape':
            loss = MAPE_mask

        else:
            raise ValueError
        return loss

    def train(self):
        self.model.train()
        best_model = None
        best_loss = float('inf')
        not_improve_count = 0
        start_time = time.time()

        for epoch in range(self.epoch_num):
            total_loss = 0
            self.model.train()

            for batch_idx ,(data,target,lens,speed_target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                pred = self.model(data,self.adj_mx,lens,self.scaler)
                pred = pred.cuda()
                target = target.cuda()
                loss = self.loss(pred,target)
                loss.backward()
                # for name, parms in self.model.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
                # return
                self.optimizer.step()
                total_loss += loss.item()

                if batch_idx % self.config.log_step == 0:
                    self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                        epoch,batch_idx,self.train_per_epoch,loss.item()))

                torch.cuda.empty_cache()

            epoch_average_loss = total_loss/self.train_per_epoch
            self.logger.info("*"*10+"Train Epoch {}: average loss: {:.6f}".format(epoch,epoch_average_loss))

            if epoch_average_loss >1e6:
                self.logger.warn('The gradient is exploded,Ending!')
                break

            #val for each epoch
            val_loss = self.val_epoch(epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                not_improve_count = 0
                self.logger.info('Current is the best model! Saving...')
                best_model = copy.deepcopy(self.model.state_dict())

            else:
                not_improve_count += 1
                print("loss has no prove for {} epoch...".format(not_improve_count))

            #judge if need early stop
            if self.config.early_stop and self.config.early_stop_patience <= not_improve_count:
                self.logger.info("the loss have not improved for {} epochs,stop trainning! ".format(
                    self.config.early_stop_patience))
                break

        end_time = time.time()
        train_time = end_time - start_time
        self.logger.info("Toal trainning time: {} mins,the best loss in valid is {:.6f}".format(train_time/60,best_loss))

        #save best model
        if  self.args.is_train:
            torch.save(best_model,self.best_path)
            self.logger.info('The best model is save to {}'.format(self.best_path))

        #test for all metrics
        self.test(best_model)


    def val_epoch(self,epoch):
        self.model.eval()
        total_val_loss = 0
        val_len = len(self.valid_loader)
        val_mape = []
        val_rmse = []
        with torch.no_grad():
            for batch_idx ,(data,target,lens,speed_target) in tqdm(enumerate(self.valid_loader),total=val_len):
                pred = self.model(data,self.adj_mx,lens,self.scaler)
                target = target.cuda()
                loss = self.loss(pred,target)
                val_mape.append(MAPE_mask(pred,target).item())
                val_rmse.append(RMSE_mask(pred,target).item())
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
            val_average_loss = total_val_loss / val_len
            val_mape = np.mean(val_mape)
            val_rmse = np.mean(val_rmse)
            self.logger.info('*'*10+'Val Epoch {}: average loss: {:.4f}, mape:{:.4f} ,rmse:{:.4f}'.format(epoch,val_average_loss,val_mape,val_rmse))

        return val_average_loss

    def test(self,best_path,train=True):
        if train:
            self.model.load_state_dict(best_path)
        else:
            self.model.load_state_dict(torch.load(best_path))
        self.model.eval()
        total_loss = 0
        y_pred = []     #batch_num*[order_num(每个batch不一样),1]
        y_label = []
        with torch.no_grad():
            for batch_idx,(data,target,lens,speed_target) in tqdm(enumerate(self.test_loader),total=len(self.test_loader)):
                pred = self.model(data,self.adj_mx,lens,self.scaler).cuda()      #order_num,1
                target = target.cuda()              #order_num,1
                y_pred.append(pred)
                y_label.append(target)
            mae,rmse,mape =All_Metrics_dynamic(y_pred,y_label)
            self.logger.info("Average  MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.2f}%".format(
                mae, rmse, mape * 100))





