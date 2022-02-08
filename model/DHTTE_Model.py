import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_sparse import spmm
from torch_geometric.data import Data
from model.util.layers import NormLayer,Time_embedding_layer
from torchstat import stat
from lib.gpu_mem_track import MemTracker

class dy_gat(nn.Module):
    """
    my gat for sparse
    """
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2,time_dim=8,concat = False):
        super(dy_gat, self).__init__()
        #c_in = (order * support_len + 1) * c_in
        self.mlp = nn.Linear(c_in,c_out,bias=True)
        self.dropout = dropout
        self.order = order
        self.time_embed_layer = nn.Linear(7+12+24,time_dim,bias=False)
        self.atten_pool = nn.Parameter(torch.zeros(time_dim,2*c_out))
        self.leakyrelu = nn.LeakyReLU()
        self.concat = concat
        nn.init.xavier_normal(self.atten_pool,gain=1.414)


    def forward(self,x,timeoh,support):
        """
        x is dense tensor (B,T,N,F)
        timeoh is the onehot of the time （B,43）
        support is sparse tensor
        """
        B,T,N,F_dim = x.shape
        if support.is_sparse:
            edge = support._indices()
        else:
            edge = support

        h = self.mlp(x)  #(B,T,N,F_out)

        time_emb = self.time_embed_layer(timeoh)
        attn = torch.mm(time_emb,self.atten_pool)  #(B,2*F_out)

        if support.is_sparse:
            edge_h = torch.cat((h[:,:,edge[0,:],:],h[:,:,edge[1,:],:]),dim=-1)      #(B,T,E,2*F_out)
            values = torch.einsum('btnc,bc->btn',(edge_h,attn)).contiguous()    #(B,T,E)
        else:
            edge_h = torch.cat([h.repeat(1,1,1,N).view(B,T,N*N,-1),h.repeat(1,1,N,1)],dim=-1).view(B,T,N,N,-1)
            values = torch.einsum('btnvf,bf->btnv',(edge_h,attn)).contiguous()

        edge_prob = self.leakyrelu(values)  #(B,T,E)
        edge_e = torch.exp(edge_prob-torch.max(edge_prob,dim=2,keepdim=True)[0])
        e_rowsum = spmm(edge,edge_e,m=N,n=N,matrix=torch.ones(size=(B,T,N,1)).cuda())   #(B,T,N,1)
        #edge_e = F.dropout(edge_e,self.dropout,training=self.training)  #detele it tomorrow
        h_prime = spmm(edge,edge_e,m=N,n=N,matrix=h)
        h_prime = h_prime.div(e_rowsum + torch.Tensor([9e-15]).cuda())
        h_prime = F.dropout(h_prime,self.dropout,training = self.training)
        if self.concat:
            h_prime = F.elu(h_prime)
        # else:
        #     h_prime = F.elu(h_prime)
        #     h_prime = F.log_softmax(h_prime,dim=1)
        return h_prime

class Temporal_conv(nn.Module):
    def __init__(self,input_dim,output_dim,kernel_size = 2,residual=False):
        super(Temporal_conv,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.filter = nn.Conv2d(input_dim, output_dim, (1, kernel_size))
        self.gate = nn.Conv2d(input_dim, output_dim, (1,kernel_size))
        self.receptive_field = kernel_size

    def forward(self,x):
        """
        :param x: B,T,N,C_in
        :return: B,T-kt+1,N,C_out
        """
        b,l,n,c = x.shape
        x = x.permute(0,3,2,1)  #B,F,N,T
        if  l < self.receptive_field:
            x = F.pad(x,(self.receptive_field-l,0,0,0))

        gate = F.sigmoid(self.gate(x))
        filter = F.tanh(self.filter(x))
        x = filter * gate
        x = x.permute(0,3,2,1)
        return x

class Pre_layer(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Pre_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Conv2d(in_dim,out_dim,kernel_size=(1,1),stride=(1,1))
    def forward(self,x):
        """
        :param x: [B,seq_len=1,num_nodes,input_dim]
        :return:
        """
        x = x.transpose(1,3)
        x = self.mlp(x)
        x = x.transpose(1,3)
        x = F.sigmoid(x)
        return x


class dy_multi_gat(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=8,order=2,time_dim=8,head=0,num_nodes = 0,concat = True ,adp=True):
        super(dy_multi_gat, self).__init__()

        if head == 0:
            self.head = 1
            self.add_out = False
        else:
            self.head = head
            self.add_out = True
        self.c_out = c_out

        self.dropout = dropout
        self.attention = nn.ModuleList(dy_gat(c_in,c_out,
                                              dropout,
                                              support_len=support_len,
                                              order=order,
                                              time_dim=time_dim,concat=concat) for _ in range(self.head))
        self.attention_list = nn.ModuleList(self.attention for _ in range(support_len))
        self.out_layer = st_gat(c_out*self.head,c_out,dropout,support_len=support_len,order=order,time_dim=time_dim,concat=False)
        self.fussion_layer = nn.Linear((support_len+2)*c_out,c_out) if adp else nn.Linear((support_len+1)*c_out,c_out)

        #adp
        self.adp = adp
        self.node_emb = 32
        self.node2vec = nn.Parameter(torch.randn(num_nodes,self.node_emb),requires_grad=True)
        self.adp_gconv = gcn(c_in,c_out,dropout,support_len=1,order=2,static=True)
        nn.init.xavier_normal(self.node2vec,gain=1.414)


        #time_emb
        self.time_embed_layer = nn.Linear(7 + 12 + 24, time_dim)

        #atten fuse
        self.w_pool = nn.Linear(time_dim,c_out)
        self.W_trans = nn.Linear(c_out,6*c_out)
        self.Wk = nn.Linear(c_out,1)


        self.atten_size = c_out*6
        self.query = nn.Linear(c_out, c_out*6)
        self.key = nn.Linear(c_out, c_out*6)
        self.value = nn.Linear(c_out, c_out*6)
        self.dense = nn.Linear(c_out*6, c_out)







    def forward(self,x,timeoh,support_list):
        """
        x is dense tensor (B,T,N,F)
        timeoh is the onehot of the time （B,43）
        support is sparse tensor list
        """
        B,T,N,_ = x.shape
        support_len = len(support_list)+2
        out = x
        for i in range(len(support_list)):
            support = support_list[i].cuda()
            y = torch.cat([att(x,timeoh,support) for att in self.attention_list[i]],dim=-1)

            if out == None:
                out = y
            else:
                out = torch.cat((out,y),dim=-1)

        #adp gconv
        if self.adp:
            adp_adj = torch.mm(self.node2vec,self.node2vec.t())
            adp_adj = F.softmax(F.relu(adp_adj), dim=1)
            x_adp = self.adp_gconv(x.transpose(1,3),[adp_adj]).transpose(1,3)
            out = torch.cat([out,x_adp],dim=-1) # B,T,N,F
        out = out.reshape(B,T,N,support_len,self.c_out).contiguous() #B,T,N,K,F


        #att_v2_self_att
        query = self.query(out)
        key = self.key(out)
        value = self.value(out)
        attention_score = torch.matmul(query,key.transpose(-1,-2))  #B,T,N,K,K
        attention_score = attention_score / math.sqrt(self.atten_size)
        attention_score = F.softmax(attention_score,dim=-1)
        out = torch.matmul(attention_score,value)
        #out = out.reshape(B, T, N, -1)
        out = self.dense(out)
        out = torch.mean(out,dim=-2)


        #out= self.fussion_layer(out)
        out = F.elu(out)
        return out


class ST_block(nn.Module): #add skip and adp
    def __init__(self,num_nodes,support_len,in_channel,hid_chennel,end_channel,kernel = 2,time_dim=8,drop_out=0.3,heads_num=0,order=1):
        super(ST_block, self).__init__()
        self.num_nodes = num_nodes
        self.support_len = support_len
        self.dropout = drop_out
        self.start_tconv = Temporal_conv(in_channel,hid_chennel,kernel_size=kernel)
        self.end_tconv = Temporal_conv(hid_chennel,end_channel,kernel_size=kernel)
        self.dygconv = dy_multi_gat(hid_chennel,hid_chennel,drop_out,support_len=support_len,
                                    time_dim=time_dim,order = order,num_nodes=num_nodes,concat=True)
        self.norm_layer = nn.BatchNorm2d(end_channel)
        self.skip_conv = nn.ModuleList()
        self.skip_conv.append(nn.Conv1d(hid_chennel, end_channel, kernel_size=(1, 1)))
        self.skip_conv.append(nn.Conv1d(end_channel, end_channel, kernel_size=(1, 1)))

    def forward(self,x,timeoh,support_list):
        """
        :param x: [batch_size,seq_len,num_nodes,input_dim]
        :return: output: [batch_size,seq_len-2*(kt-1),num_nodes,output_dim]
        """
        x = self.start_tconv(x)
        s = self.skip_conv[0](x.transpose(1,3))  #B,D,N,T
        skip = s.transpose(2,3).reshape([s.shape[0],-1,s.shape[2],1]).contiguous()

        x = self.dygconv(x,timeoh,support_list)
        x = self.end_tconv(x)
        x = x.transpose(1,3)

        s = self.skip_conv[1](x)
        skip = torch.cat([skip,s.transpose(2,3).reshape([s.shape[0],-1,s.shape[2],1]).contiguous()],dim=1)  #(B,F*T,N,1)


        x = self.norm_layer(x)
        x = x.transpose(1,3)
        return x,skip

class ST_block_tc(nn.Module): #add skip and adp
    def __init__(self,num_nodes,support_len,in_channel,hid_chennel,end_channel,kernel = 2,time_dim=8,drop_out=0.3,heads_num=0,order=1):
        super(ST_block_tc, self).__init__()
        self.num_nodes = num_nodes
        self.support_len = support_len
        self.dropout = drop_out
        self.start_tconv = Temporal_conv(in_channel,hid_chennel,kernel_size=kernel)
        self.end_tconv = Temporal_conv(hid_chennel,end_channel,kernel_size=kernel)
        self.dygconv = dy_multi_gat(hid_chennel,hid_chennel,drop_out,support_len=support_len,
                                    time_dim=time_dim,order = order,num_nodes=num_nodes,concat=True)
        self.norm_layer = nn.BatchNorm2d(end_channel)
        self.skip_conv = nn.ModuleList()
        self.skip_conv.append(nn.Conv1d(2*hid_chennel, end_channel, kernel_size=(1, 1)))
        self.skip_conv.append(nn.Conv1d(2*end_channel, end_channel, kernel_size=(1, 1)))

        self.t_conv = nn.ModuleList()
        self.t_conv.append(tgcn(hid_chennel, hid_chennel, drop_out, support_len=1, order=1))
        self.t_conv.append(tgcn(end_channel, end_channel, drop_out, support_len=1, order=1))



    def forward(self,x,timeoh,support_list,t_adp):
        """
        :param x: [batch_size,seq_len,num_nodes,input_dim]
        :return: output: [batch_size,seq_len-2*(kt-1),num_nodes,output_dim]
        """
        x = self.start_tconv(x)
        #residual = x.transpose(1, 3)
        x_tconv = self.t_conv[0](x.transpose(1,3),t_adp[0])

        s = torch.cat([x.transpose(1,3),x_tconv],dim=1)
        s = self.skip_conv[0](s)  #B,D,N,T
        skip = s.transpose(2,3).reshape([s.shape[0],-1,s.shape[2],1]).contiguous()

        x = self.dygconv(x,timeoh,support_list)
        x = self.end_tconv(x).transpose(1,3)
        x_tconv = self.t_conv[1](x,t_adp[1])

        s = torch.cat([x,x_tconv],dim=1)
        s = self.skip_conv[1](s)
        skip = torch.cat([skip,s.transpose(2,3).reshape([s.shape[0],-1,s.shape[2],1]).contiguous()],dim=1)  #(B,F*T,N,1)


        #x  = x + residual[:,:,:,-x.size(3):]
        x = self.norm_layer(x)
        x = x.transpose(1,3)
        return x,skip

class HDTTE(nn.Module):
    def __init__(self,num_nodes,config,support_len):
        super(HDTTE, self).__init__()
        self.node_num = num_nodes
        self.drop_out = config.dropout
        self.device = config.device
        self.in_channel = config.dynamic_dim+config.static_dim
        self.hid_channel = config.hid_channel
        self.end_channel = config.end_channel
        self.time_emb_dim = config.time_embed_dim
        self.time_oh_dim = config.time_dim
        self.static_dim = config.static_dim
        self.days = config.days
        self.recent = config.recent
        self.weeks = config.weeks
        self.order = config.max_diffusion_step
        self.input_len = self.recent+self.days+self.weeks
        self.timetype,self.Istart,self.Iend = self.get_type_list(self.recent,self.weeks,self.days)
        self.kernel_size = 2
        self.block_num = config.block_num


        self.ST_Conv_block = nn.ModuleList()
        self.ST_Last_Layer = nn.ModuleList()

        t_emb_dim = 32
        self.t_dim_list = [3,2,1,1]
        self.t_dict_n1 = {}
        self.t_dict_n2 = {}
        self.tadp_list1 = nn.ParameterList()
        self.tadp_list2= nn.ParameterList()
        for i in self.t_dim_list:
            adp1  = nn.Parameter(torch.randn(i,num_nodes,t_emb_dim),requires_grad=True)
            adp2 = nn.Parameter(torch.randn(i,t_emb_dim,num_nodes),requires_grad=True)
            nn.init.xavier_normal(adp1,gain = 1.414)
            nn.init.xavier_normal(adp2, gain=1.414)
            self.tadp_list1.append(adp1)
            self.tadp_list2.append(adp2)

        dims = 40
        self.nodevec_fuse_recent = nn.Parameter(torch.randn(num_nodes, dims).to(self.device), requires_grad=True).to(self.device)
        self.nodevec_fuse_day = nn.Parameter(torch.randn(num_nodes, dims).to(self.device), requires_grad=True).to(
            self.device)

        self.fusion_conv = gcn(c_in=77,c_out=77,support_len=1,order=2,dropout=self.drop_out,static=True)

        for type in self.timetype:
            self.ST_Blocks = nn.ModuleList([ST_block_tc(num_nodes=num_nodes, support_len=support_len, in_channel=self.in_channel,
                             hid_chennel=self.hid_channel, end_channel=self.end_channel,
                             time_dim=self.time_emb_dim, drop_out=self.drop_out,
                             order=self.order, kernel=self.kernel_size)])
            for i in range(1,self.block_num):
                self.ST_Blocks.append(ST_block(num_nodes=num_nodes, support_len=support_len, in_channel=self.end_channel,
                                 hid_chennel=self.hid_channel, end_channel=self.end_channel,
                                 time_dim=self.time_emb_dim, drop_out=self.drop_out,
                                 order=self.order, kernel=self.kernel_size))
            self.ST_Conv_block.append(self.ST_Blocks)
            self.ST_Last_Layer.append(nn.Conv2d(in_channels=77,out_channels=self.end_channel,
                                               kernel_size=(1,1)))

        self.pred_layer = Pre_layer(in_dim=len(self.timetype)*self.end_channel,out_dim=1)
        self.end_conv = nn.Conv2d(in_channels=len(self.timetype) * 77,out_channels=len(self.timetype) * 128,kernel_size=(1,1))
        self.pred_layer_skip1 = Pre_layer(in_dim=77, out_dim=1)
        self.pred_layer_skip2 = Pre_layer(in_dim=len(self.timetype) * 128, out_dim=1)
        self.norm = nn.BatchNorm2d(77)


    def dtconstruct(self,  source_embedding, target_embedding):
        adp = torch.einsum('twd, tdv->twv', source_embedding, target_embedding)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def forward(self,input,support_list,lens,scaler):
        """
        :param input: input： (batch_fes, batch_link_move_sp,lens)    batch_sum_time
        batch_fes:(batch_size,input_len,num_nodes,feature_dim) 包含了recent，day，和week
        batch_link_move_sp:(batch_size,order_nums,num_nodes)   代表这个batch中每个订单经过的node
        batch_sum_time:(order_nums,1) #代表每个订单的总时间
        :return:
        """
        t_adp = []
        for i in range(len(self.t_dim_list)):
            adp = self.dtconstruct(self.tadp_list1[i],self.tadp_list1[i].transpose(1,2))
            t_adp.append(adp)

        x,traj,link_id = input
        x = x.float().cuda()
        traj = traj.cuda()
        x_dy,time_oh,x_static = x.split([1,self.time_oh_dim,self.static_dim],dim=-1)
        time_oh = time_oh[:,0,0,:]
        x = torch.cat([x_dy,x_static],dim=-1)


        fusion_adp_day = [F.softmax(F.relu(torch.mm(self.nodevec_fuse_day,self.nodevec_fuse_day.t())),dim=1)]
        fusion_adp_recent = [F.softmax(F.relu(torch.mm(self.nodevec_fuse_recent, self.nodevec_fuse_recent.t())), dim=1)]
        out = None
        for idx,timetype in enumerate(self.timetype):
            skip_type = None
            x_type = x[:,self.Istart[timetype]:self.Iend[timetype],:,:]
            for i in range(self.block_num):
                if i == 0:
                    x_type,skip = self.ST_Conv_block[idx][i](x_type,time_oh,support_list,t_adp[i*2:i*2+2])
                else:
                    x_type, skip = self.ST_Conv_block[idx][i](x_type, time_oh, support_list)
                if skip_type == None:
                    skip_type = skip
                else:
                    skip_type = torch.cat([skip_type, skip], dim=1)

            skip = skip_type
            if out == None:
                out = self.fusion_conv(skip,fusion_adp_day)#+skip
            else:
                out = out + self.fusion_conv(skip,fusion_adp_recent)#+skip



        out = out.transpose(1,3)
        out = self.pred_layer_skip1(out).view(-1,1,self.node_num)     #B,1,N,D
        out = out * 35
        out = torch.div(traj,out)
        out = out.sum(2)
        return out

    def get_type_list(self, recents, weeks, days):
        if recents:
            types = ["recent"]
        else:
            types = []
        if days:
            types += ["days"]
        if weeks:
            types += ["weeks"]
        Istart = {"weeks":0, "days":weeks, "recent":weeks+days}
        Iend = {"weeks":weeks, "days":weeks+days, "recent":weeks+recents+days}
        assert len(types) > 0
        return types,Istart,Iend




#util
class tnconv(nn.Module):
    def __init__(self):
        super(tnconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bcvl,lwv->bcwl', (x, A))
        return x.contiguous()

class dy_tnconv(nn.Module):
    def __init__(self):
        super(dy_tnconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bcvl,blwv->bcwl', (x, A))
        return x.contiguous()

class tgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2,static = True):
        super(tgcn, self).__init__()
        self.tnconv = tnconv() if static else dy_tnconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]

        x1 = self.tnconv(x, support)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.tnconv(x1, support)
            out.append(x2)
            x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        #h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2,static = False):
        super(gcn, self).__init__()
        if static:
            self.nconv = nconv_static()
        else:
            self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return x.contiguous()

class nconv_static(nn.Module):
    def __init__(self):
        super(nconv_static, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,wv->ncwl', (x, A))
        return x.contiguous()


