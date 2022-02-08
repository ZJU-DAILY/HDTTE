import torch
import os
import time
import torch.nn as nn
import argparse
import random
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from lib.dataloader import MyDataset
from model.Trainer import *
from model.config import Config,Config_BJ
from lib import support,metrics,logger
import model.DHTTE_Model as bst





#parser
parser = argparse.ArgumentParser(description='argument')
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--mode',default='train',type=str)
parser.add_argument('--is_train',action='store_true')
parser.add_argument('--seed',default=10,type=int)
parser.add_argument('--batch_size',default=32,type=int)
parser.add_argument('--data',default='BJ',type=str)
parser.add_argument('--marks',default='',type=str)
parser.add_argument('--useadj',default=2,type=int)
parser.add_argument('--block',default=2,type=int)
args = parser.parse_args()


#get config
if args.data =='WUHAN':
    config = Config()
elif args.data =='BJ':
    config = Config_BJ()
config.batch_size = args.batch_size
config.block_num = args.block

#init seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#set gpu device
device = torch.device(args.device if torch.cuda.is_available() else "cpu")


#config log path
current_time = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
current_log = config.log_dir
current_log = os.path.join(current_log,current_time)
config.log_dir = current_log

# get support
data_dir = config.dataset_dir
adj_mx = {}
use_adj = {1:["total_adj.mat"],2:["adj.mat", "total_adj.mat"]}
if args.data == "WUHAN":
    adj_files = use_adj[args.useadj]
elif args.data =="BJ":
    adj_files = use_adj[args.useadj]
road_adj = []
for file in adj_files:
    adj_path = os.path.join(data_dir, file)
    print("loading adj matrix: ", adj_path)
    adj_mx_file = sio.loadmat(adj_path)
    suffix = file[-7:-4]
    for k in adj_mx_file:
        if k in ['__header__', '__version__', '__globals__']:
            continue
        if k not in adj_mx:
            adj_mx[k + suffix] = adj_mx_file[k]
        else:
            raise Exception("There is already a %s adj matrix" % k)
for k, v in adj_mx.items():
    road_adj.append(v.toarray())

support_list = []
for i in road_adj:
    support_list.append(torch.from_numpy(i).to_sparse())



#build model and init
model = bst.HDTTE(num_nodes=config.num_nodes, config=config, support_len=len(support_list))
model.to(device)
#get trainner
if args.multitask == False:
    trainer = trainer(args,config,model,support_list)
else:
    trainer = trainer_mul(args, config, model, support_list)
if args.mode == 'train':
    trainer.train()
if args.mode == 'test':
    best_path = './experiment/WUHAN/2022-01-13 21:05:02/best_model.pth'
    trainer.test(best_path,train=False)




