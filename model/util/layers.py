import torch
import torch.nn as nn
import torch.nn.functional as F


class NormLayer(nn.Module):
    def __init__(self,n,dim):
        super(NormLayer, self).__init__()
        self.n = n
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones([1,1,n,dim]))
        self.beta = nn.Parameter(torch.zeros([1,1,n,dim]))
        #self.normlayer = nn.InstanceNorm2d()

    def forward(self,input):
        batch_size,seq_len,n,dim = list(input.shape)
        assert (self.n == n)
        assert (self.dim == dim)
        x =F.instance_norm(input,weight=self.gamma,bias=self.beta,eps=1e-06)
        return x


class Time_embedding_layer(nn.Module):
    def __init__(self,time_dim,embed_dim):
        super(Time_embedding_layer, self).__init__()
        self.time_dim = time_dim
        self.embed_dim = embed_dim
        self.emblayer = nn.Linear(self.time_dim,self.embed_dim)

    def forward(self,input):
        """
        :param input: [B*seq,N,D]     [dynamic+static,D-time_dime:D]  is the time_feature
        :return: input_without_time(B*seq,N,D-time_dim),
                    time_embedding(B*seq,time_embed_dim)
        """
        input_without_time = input[:,:,:-self.time_dim]
        time_fes = input[:,1,-self.time_dim:]  #[B*seq,time_dim]
        time_embed = self.emblayer(time_fes)
        return input_without_time,time_embed

def get_graph_info(data):
    print('** G attrs: ', '\n', data.keys)
    print('** G node features: ', '\n', data.x)
    print('** G node num: ', '\n', data.num_nodes)
    print('** G edge num: ', '\n', data.num_edges)
    print('** G node feature num: ', '\n', data.num_node_features)
    print('** G isolated nodes: ', '\n', data.contains_isolated_nodes())
    print('** G self loops: ', '\n', data.contains_self_loops())
    print('** G is_directed: ', '\n', data.is_directed())



if __name__ == '__main__':
    norm = NormLayer(100,10)
    for name,w in norm.named_parameters():
        print(name,w.shape)
