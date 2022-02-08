import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
from model.config import Config
from lib.scaler import StandardScaler, StaticFeatureScaler
import torch.nn.functional as F
import torch.nn as nn
import scipy.io as sio
import scipy.sparse as sp


def load_dataset(config):
    """

    :param config:
    :return: static_fes,dynamic_fea_all,all_data
    :param static_fes:(nodes_num,static_dim)
    :param dynamic_fea_all:(periods_num,nodes_num,dynamic_dim) dynamic_dim=(recent_len+day_len+week_len)
    """
    base_dir = config.dataset_dir
    node_num = config.num_nodes
    recent_len = config.seq_len
    days_len = config.days
    weeks_len = config.weeks
    seq_len = recent_len + days_len + weeks_len
    per_period = config.per_period
    static_dim = config.static_dim
    dynamic_dim = config.dynamic_dim
    time_dim = config.time_dim
    eta_label_dir = os.path.join(base_dir,'eta_label.npz')
    dynamic_fes_dir = os.path.join(base_dir,'new_dynamic_fes.npz')
    link_info_dir = os.path.join(base_dir,'link_info.npz')
    data = {}

    #load npz data
    eta_label = np.load(eta_label_dir,allow_pickle=True)
    dynamic_fes = np.load(dynamic_fes_dir)
    link_info = np.load(link_info_dir)
    print(list(eta_label.keys()),'\n',list(dynamic_fes.keys()),'\n',list(link_info.keys()))

    #static feature:
    print("the link_info shape: ",link_info['link_info'].shape)
    data['link_length'] = link_info['link_info'][:,0]
    scaler0 = StaticFeatureScaler()
    static_fes = scaler0.transform(link_info['link_info'])[:,0:static_dim]  #(node_num,static_dim)
    print("static_fes info:",static_fes.shape)

    #dynamic feature:
    dynamic_info,dynamic_period = dynamic_fes['fes'],dynamic_fes['periods']
    dynamic_time_info = dynamic_info[...,dynamic_dim:dynamic_dim+time_dim]
    dynamic_info = dynamic_info[...,:dynamic_dim]
    scaler1 = StandardScaler(dynamic_info.mean(),dynamic_info.std())
    dynamic_info = scaler1.transform(dynamic_info)
    dynamic_info = np.concatenate([dynamic_info,dynamic_time_info],axis=-1)

    #concate the dynamic_feature of recent,days,weeks
    periods_num_day = int(24*60/per_period)
    all_periods_num = len(dynamic_period)
    dynamic_fea_all =[]
    speed_label = []
    if weeks_len != 0:
        for period in range(all_periods_num):
            if period - recent_len < 0:
                recent_idx = [period for _ in range(recent_len)]
            else:
                recent_idx = [i for i in range(period - recent_len, period)]
            if period - periods_num_day * days_len < 0:
                day_idx = [period for _ in range(days_len)]
            else:
                day_idx = [i for i in range(period - periods_num_day * days_len, period, periods_num_day)]
            if period - periods_num_day*weeks_len*7 < 0:
                week_idx = [period for _ in range(weeks_len)]
            else:
                week_idx = [i for i in range(period-periods_num_day * weeks_len*7,period,periods_num_day*7)]
            all_idx = week_idx + day_idx + recent_idx
            all_fes = dynamic_info[all_idx]
            all_fes = all_fes.reshape(time_dim+dynamic_dim, node_num, recent_len+days_len+weeks_len)
            dynamic_fea_all.append(all_fes)
    elif weeks_len == 0 and days_len != 0 :
        for period in range (all_periods_num):
            if period + recent_len > all_periods_num:   #for label construct
                label_idx = [period for _ in range(recent_len)]
            else:
                label_idx = [i for i in range(period,period+recent_len)]

            if period - recent_len < 0:
                recent_idx = [period for _ in  range(recent_len)]
            else:
                recent_idx = [i for i in range(period-recent_len,period)]

            if period - periods_num_day*days_len < 0:
                day_idx = [period for _ in range(days_len)]
            else:
                day_idx = [i for i in range(period-periods_num_day*days_len,period,periods_num_day)]
            all_idx = recent_idx + day_idx
            all_fes = dynamic_info[all_idx]
            all_fes = all_fes.transpose(2,1,0)
            dynamic_fea_all.append(all_fes)

            speed_label.append(dynamic_info[label_idx])
    dynamic_fea_all = np.stack(dynamic_fea_all,axis=0)
    dynamic_fea_all = dynamic_fea_all.transpose(0,3,2,1)
    speed_label = np.stack(speed_label,axis=0)[...,:1]
    speed_label = scaler1.inverse_transform(speed_label,tensor=False)
    print("dynamic_all_info: ",dynamic_fea_all.shape)



    #deal the eta_label
    mode_list = ['train','valid','test']
    all_data ={}
    for mode in mode_list:
        eta_label_train = eta_label[mode]
        eta_label_train_period = eta_label['%s_periods'%mode]
        print("load eta_label_%s, shape:%s"%(mode,eta_label_train.shape))

        period_order = []       #
        for period in range(len(eta_label_train_period)):
            orders_num  = len(eta_label_train[period])
            period_sum_time = []
            period_link_id = []
            a_period_order = np.zeros((orders_num,node_num), dtype=float)               #(orders_num,node_num)
            for order_idx,(link_idxs,link_moves,time_spent) in enumerate(eta_label_train[period]):
                period_link_id.append(link_idxs)
                for i,link_id in enumerate(link_idxs):
                    a_period_order[order_idx][link_id] += link_moves[i]
                period_sum_time.append(time_spent)
            period_sum_time = np.array(period_sum_time)
            period_order.append((a_period_order,period_sum_time,period_link_id))


        period_real_idx = [i for i in eta_label_train_period]
        period2batch = dict(zip(period_real_idx,period_order))
        all_data[mode] = period2batch

    mix_fea = np.expand_dims(static_fes, axis=0).repeat(seq_len, axis=0)
    mix_fea = np.expand_dims(mix_fea,axis=0).repeat(all_periods_num,axis=0)
    mix_fea = np.concatenate((dynamic_fea_all,mix_fea),axis=-1)  #(periods_num,seq_len,nodes_num,dynamic_dim+static_dim)
    print("the shape with dynamic and static info: ",mix_fea.shape)
    return mix_fea,all_data,scaler1,speed_label         #mix_fes : dynamic_fes is before static  [dynamic+time+static]


class MyDataset(Dataset):
    def __init__(self,config,train=True,test=False,valid=False):
        self.train = train
        self.test = test
        self.valid = valid
        self.config = config
        self.mix_fea,self.all_data,self.scaler,self.speed_label = load_dataset(config)

        if self.train:
            self.node_fea = self.mix_fea[config.train_periods[0]:config.train_periods[1]]
            self.eta_data = self.all_data['train']          # #period2batch {period_id:(a_period_order,period_sum_time),...}
            self.period_start_idx = config.train_periods[0]
        elif self.test:
            self.node_fea = self.mix_fea[config.test_periods[0]:config.test_periods[1]]
            self.eta_data = self.all_data['test']
            self.period_start_idx = config.test_periods[0]
        else:
            self.node_fea = self.mix_fea[config.valid_periods[0]:config.valid_periods[1]]
            self.eta_data = self.all_data['valid']
            self.period_start_idx = config.valid_periods[0]

        self.node_fea = torch.from_numpy(self.node_fea)
        self.speed_label = torch.from_numpy(self.speed_label)
        for k,v in self.eta_data.items():       #turn the eta array to tensor
            v1,v2,link_id = v
            v1 = torch.from_numpy(v1)
            v2 = torch.from_numpy(v2)
            self.eta_data[k] = (v1,v2,link_id)

        self.size = self.node_fea.shape[0]
        print(self.node_fea.shape)      #(100,12,300,9)


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        real_period_idx = idx + self.period_start_idx
        the_period_order,period_sum_time,link_id = self.eta_data[real_period_idx]
        the_period_feature = self.node_fea[idx].squeeze()
        data = (the_period_feature,the_period_order,link_id)
        eta_label = period_sum_time
        speed_label = self.speed_label[real_period_idx]
        return data,eta_label,speed_label


def collate_fn(batch):
    data,label,speed_label = zip(*batch)
    data,label,speed_label = list(data),list(label),list(speed_label)                 #tensor list (ordernum,roadnum)
    fea,order,traj = list(map(list,zip(*data)))
    lens = np.asarray([len(item) for item in label ])

    max_order_num = lens.max()
    pad_num = list(map(lambda x:max_order_num-x,lens))
    for idx in range(len(data)):
        order[idx] = F.pad(order[idx],[0,0,0,pad_num[idx]])
        label[idx] = F.pad(label[idx],[0,pad_num[idx]])

    fea = torch.stack(fea,dim=0)
    order = torch.stack(order,dim=0)
    label = torch.stack(label,dim=0)
    speed_label = torch.stack(speed_label,dim=0)
    return (fea,order,traj),label,lens,speed_label  #B,T,N,D






