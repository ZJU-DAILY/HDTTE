import numpy as np
import torch

class Config(object):

    #data
    log_dir = './experiment/WUHAN'
    dataset_dir = '../Data/Wuhan'
    train_data_root = ''
    test_data_root = ''
    load_model_path = ''
    best_model_path = ''
    device = 'cuda:0'
    static_dim = 6
    dynamic_dim = 1
    per_period = 10

    #model
    model_dir = ''
    model = 'HDTTE'
    max_diffusion_step = 2
    temporal_kernal_size = 2
    block_num = 2
    num_nodes = 2626
    num_rnn_layer = 1
    hidden_dim = 8
    output_dim = 11
    seq_len = 4
    days = 4
    weeks = 0
    road_net_num = 7
    heads_num = 1
    time_dim = 43      #12months + 7 days + 24 hours
    time_embed_dim = 8
    hid_channel = 8     #8
    end_channel = 11    #11
    recent = 4

    #train

    train_periods = [0, 518]
    valid_periods = [518, 691]
    test_periods = [691, 864]
    batch_size = 32
    seed = 10
    base_lr = 0.01
    lr_decay_ratio = 0.15
    lr_decay_epoch = 35
    weight_decay = 0
    dropout =  0.0          #0.0
    epochs =  300
    epsilon =  0.005            #1e-8
    max_grad_norm = 5
    optimizer = 'amsgrad'
    patience = 12
    threshold = 3
    regular_rate = 0.0001
    gpu_num = 1
    loss_func = 'mae'
    early_stop =True
    early_stop_patience = 30

    #log
    log_step = 1


class Config_BJ(object):

    #data
    log_dir = './experiment/BJ'
    dataset_dir = '../Data/BJ'
    train_data_root = ''
    test_data_root = ''
    load_model_path = ''
    best_model_path = ''
    device = 'cuda:0'
    static_dim = 6
    dynamic_dim = 1
    per_period = 10

    #model
    model_dir = ''
    model = 'HetETA'
    max_diffusion_step = 2
    temporal_kernal_size = 2
    block_num = 2
    num_nodes = 3008
    num_rnn_layer = 1
    hidden_dim = 8
    output_dim = 11
    seq_len = 4
    days = 4
    weeks = 0
    road_net_num = 7
    heads_num = 1
    time_dim = 43      #12months + 7 days + 24 hours
    time_embed_dim = 8
    hid_channel = 8     #8
    end_channel = 11    #11
    recent = 4

    #train

    train_periods = [0, 777]
    valid_periods = [777, 1036]
    test_periods = [1036, 1296]
    batch_size = 32
    seed = 10
    base_lr = 0.01
    lr_decay_ratio = 0.15
    lr_decay_epoch = 35
    weight_decay = 0
    dropout =  0.0
    epochs =  200
    epsilon =  0.005            #1e-8
    max_grad_norm = 5
    optimizer = 'amsgrad'
    patience = 12
    threshold = 3
    regular_rate = 0.0001
    gpu_num = 1
    loss_func = 'mae'
    early_stop =True
    early_stop_patience = 20

    #log
    log_step = 1

