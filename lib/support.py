# encoding=utf-8
import numpy as np
import scipy
import scipy.sparse as sp
import functools



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        # to be order for sparse tensor!
        shape = mx.shape
        flatten = float(shape[-1])*mx.row + mx.col
        order_indices = np.argsort(flatten)
        coords = np.vstack((mx.row[order_indices], mx.col[order_indices])).transpose()
        values = mx.data[order_indices]
        return coords.astype(np.int64), values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx



def calculate_normalized_laplacian(adj, to_tuple=True):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj: a symetric matrix
    :param to_tuple: bool
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    if to_tuple:
        normalized_laplacian = sparse_to_tuple(normalized_laplacian)
    return normalized_laplacian



def get_symetric_matrix(adj):
    adj_mx = adj + adj.transpose()
    rep = np.where(adj_mx.data > 1)
    adj_mx.data[rep] = 1.
    return adj_mx


def sort_key(x,y):
    if "op5" in x:
        return 1
    else:
        return -1

def get_cheb_support(adj, K=3, to_tuple=True):
    n = adj.shape[0]
    adj = get_symetric_matrix(adj)         #结合所有type，获得了邻接矩阵01矩阵
    L = calculate_normalized_laplacian(adj, to_tuple=False)
    supports = [sp.eye(n), L]
    for i in range(K-2):
        Li = 2 * (supports[-1]).dot(L).tocoo() - supports[-2]
        supports.append(Li)
    if to_tuple:
        out = list(map(sparse_to_tuple, supports))
    else:
        out = supports
    return out

def weight_attention(cheb_adj, attention_adjs):
    num = len(attention_adjs)
    for i in range(num):
        attention_adjs[i] = attention_adjs[i].multiply(cheb_adj)
    return attention_adjs


def get_model_cheb_support(adj, K=3):           #adj：路网类型图和车辆轨迹的频率图
    supports1 = None
    supports2 = None
    atten_supports = []
    flag = 0
    adj_keys = adj.keys()
    print(adj_keys)
    adj_keys = sorted(adj_keys, key=functools.cmp_to_key(sort_key))
    print(adj_keys)
    print("keys in matrix", adj_keys)
    for k in adj_keys:
        print(k)
        if k in ['__header__', '__version__', '__globals__']:
            continue
        if not flag:
            supports1 = adj[k]
            flag = 1
        elif "adj" in k:
            supports1 += adj[k]
        else:
            if flag == 1:
                supports2 = adj[k]
                flag = 2
            else:
                supports2 += adj[k]
        #atten_supports.append(sparse_to_tuple(adj[k]))
        atten_supports.append(adj[k])
        if "op5" not in k:
            """the transpose matrix of this matrix has been added in preprocessing"""
            #atten_supports.append(sparse_to_tuple(adj[k].transpose()))
            atten_supports.append(adj[k].transpose())
    supports1 = get_cheb_support(supports1, K=K,to_tuple=False)
    supports2 = get_cheb_support(supports2, K=K,to_tuple=False)
    supports = supports1 + supports2
    return supports, atten_supports

def load_adj(adj, max_diffusion_step=2,method="HetETA"):
        atten_supports = None
        if method in ["HetETA"]:
            supports, atten_supports = get_model_cheb_support(adj=adj, K=max_diffusion_step+1)
            print("the num of supports", len(supports))
            print("the num of atten_supports", len(atten_supports))
        else:
            supports = []
        return supports, atten_supports