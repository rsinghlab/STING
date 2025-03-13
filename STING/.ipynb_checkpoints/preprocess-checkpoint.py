#This code has been adapted from https://github.com/JinmiaoChenLab/GraphST since STING uses GraphST as the outer GNN


import os
import ot
import torch
import random
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
#from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import torch_geometric
from torch_geometric.data import Data


def permutation(feature):
    # fix_seed(FLAGS.random_seed) 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 

def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
            
    '''threshold = np.partition(distance_matrix.flatten(), (n_neighbors*n_spot))[(n_neighbors*n_spot) - 1]
    interaction = (distance_matrix<= threshold).astype('uint8')
    np.fill_diagonal(interaction, 0)'''
         
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj
    
def preprocess(adata, n_top_genes):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    
def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
    
    adata.uns['HVG_list'] = list(adata_Vars.var.index)
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    
    # data augmentation
    feat_a = permutation(feat)
    
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a    
    
def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL
    
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
    
def get_gene_graph_cluster(features, avg_neigh = 7, met = 'correlation'):
    
    
    if features.shape[0] > 20:

        pca = PCA()
        pca.fit(features.T)
        var = pca.explained_variance_ratio_
        sum = 0
        n_component = 0
        while n_component < len(var)-1:
            sum += var[n_component]
            if sum >= 0.95:
                break
            n_component+=1




        pca = PCA(n_components = n_component)
        pca_features = pca.fit_transform(features.T).T
    else:
        pca_features = features.copy()
    
    
    
    euc_expression_correlations = 1 - cdist(pca_features.T,pca_features.T, metric=met)


    mini = -10
    
    euc_expression_correlations = np.nan_to_num(euc_expression_correlations, mini)

    for g in np.where(np.sum(features.copy(), axis = 0) == 0)[0]:
        euc_expression_correlations[g] = mini
        euc_expression_correlations[:, g] = mini

    num_genes = euc_expression_correlations.shape[0]
    euc_expression_correlations = np.maximum(euc_expression_correlations, euc_expression_correlations.T)
    threshold = np.sort(euc_expression_correlations, axis=None)[-(int(avg_neigh*num_genes))]
    if avg_neigh == 0:
        threshold = -mini
    if threshold == mini:
        threshold = mini + 1


    gene_expression_adj_matrix = (euc_expression_correlations>= threshold).astype('uint8') #0.275

            
    edge_index_inner = torch.from_numpy(gene_expression_adj_matrix).nonzero().t().contiguous()
    edge_index_inner = torch_geometric.utils.to_undirected(edge_index_inner)
    
    adj_mat = torch.tensor(euc_expression_correlations*(euc_expression_correlations>= threshold))
    #edge_attr = adj_mat[edge_index_inner[0], edge_index_inner[1]].t().float()
    
    return edge_index_inner#, edge_attr
    
