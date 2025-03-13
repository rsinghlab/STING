#This code has been partly adapted from https://github.com/JinmiaoChenLab/GraphST since STING uses GraphST as the outer GNN


import torch
from .preprocess import preprocess_adj, preprocess, construct_interaction, add_contrastive_label, get_feature, permutation, fix_seed, get_gene_graph, get_gene_graph_cluster
import time
import random
import numpy as np
from .model import Encoder
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
#from scipy.sparse.csc import csc_matrix
#from scipy.sparse.csr import csr_matrix
from torch_geometric.data import Data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scanpy as sc

class STING():
    def __init__(self, 
        adata,
        device= torch.device('cpu'),
        learning_rate=0.001,
        weight_decay=0.00,
        epochs=900, 
        #dim_input=3000,
        dim_output=64,
        random_seed = 2,
        alpha = 10,
        beta = 1
        ):
        '''\

        Parameters
        ----------
        adata : anndata
            AnnData object of spatial data.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        learning_rate_sc : float, optional
            Learning rate for scRNA representation learning. The default is 0.01.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 600.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 41.
        alpha : float, optional
            Weight factor to control the influence of reconstruction loss in representation learning. 
            The default is 10.
        beta : float, optional
            Weight factor to control the influence of contrastive loss in representation learning. 
            The default is 1.

        Returns
        -------
        The AnnData object including the new embeddings and attention scores.

        '''
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta

        fix_seed(self.random_seed)
        
        adata1 = adata.copy()
        sc.pp.highly_variable_genes(adata1, flavor="seurat_v3", n_top_genes=3000)
        depth = np.sum(adata1[:, adata1.var['highly_variable']].X)
        del adata1
        
        dim_input = self.features.shape[1]
        
        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata, dim_input)
        
        if 'adj' not in adata.obsm.keys():
            construct_interaction(self.adata)
         
        if 'label_CSL' not in adata.obsm.keys():    
            add_contrastive_label(self.adata)
           
        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata)
           
           
        km = KMeans(n_clusters=self.adata.obsm['spatial'].shape[0]//100, random_state=self.random_seed).fit_predict(self.adata.obsm['spatial'])
        self.adata.obsm['km'] = km
        
        self.km_graphs = [] 
        num_genes = self.adata.obsm['feat'].shape[1]
        spars = np.sum(self.adata.obsm['feat'] == 0)/self.adata.obsm['feat'].size
        avg_neigh = min(7, ((depth/100000)**0.5) * (0.35*(np.log(num_genes-30)-1)))

        for i in range(np.max(km) + 1):
            
            km_g = get_gene_graph_cluster(self.adata.obsm['feat'][km == i], avg_neigh = avg_neigh) #km_a
            
            self.km_graphs.append(km_g.to(self.device))
            
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.adj = self.adata.obsm['adj']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
        
        if dim_output > self.adata.obsm['feat'].shape[1]:
            dim_output = self.adata.obsm['feat'].shape[1]
        
        
        pca = PCA(n_components=dim_output, random_state=42) 
        pca_features = pca.fit_transform(self.adata.obsm['feat'].copy())

        self.inner_graph_features = []
        
        for i in range(self.features.shape[0]):
            gene_exp_i = self.features[i]
            gene_exp_i = gene_exp_i.reshape(-1,1)
            gene_edges = self.km_graphs[km[i]]
            inner_data = Data(x=gene_exp_i.float().to(self.device), edge_index=gene_edges)#, edge_attr = edge_weights)
            self.inner_graph_features.append(inner_data)
            
        self.inner_graph_features_a = []
        for i in range(self.features_a.shape[0]):
            gene_exp_i = self.features_a[i]
            gene_exp_i = gene_exp_i.reshape(-1,1)
            gene_edges = self.km_graphs[km[i]]
            inner_data = Data(x=gene_exp_i.float().to(self.device), edge_index=gene_edges)#, edge_attr = edge_weights)
            self.inner_graph_features_a.append(inner_data)
        
        
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
       self.adj = preprocess_adj(self.adj)
       self.adj = torch.FloatTensor(self.adj).to(self.device)
        
            
    def train(self):
        self.model = Encoder(self.dim_input, self.dim_output, self.graph_neigh,  self.device).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()

    
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)

        
        print('Begin to train ST data...')
        self.model.train()
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
              
            self.features_a = permutation(self.features)
            self.inner_graph_features_a = []
            
            for i in range(self.features_a.shape[0]):
                gene_exp_i = self.features_a[i]
                gene_exp_i = gene_exp_i.reshape(-1,1)
                gene_edges = self.km_graphs[self.adata.obsm['km'][i]]
                inner_data = Data(x=gene_exp_i.float().to(self.device), edge_index=gene_edges)
                self.inner_graph_features_a.append(inner_data)
  
            self.hiden_feat, self.emb, ret, ret_a, _, _, _ = self.model(self.features, self.features_a, self.adj, self.inner_graph_features,self.inner_graph_features_a)
            
            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
             
           
            self.loss_feat = F.mse_loss(self.features, self.emb)

            loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2)

            
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()

        
        print("Optimization finished for ST data!")
        
        with torch.no_grad():
            self.model.eval()
             
            self.adata.obsm['emb'] = self.model(self.features, self.features_a, self.adj, self.inner_graph_features,self.inner_graph_features_a)[0].detach().cpu().numpy()
            self.adata.uns['a1'] = self.model(self.features, self.features_a, self.adj, self.inner_graph_features,self.inner_graph_features_a)[-2]
                   
                
            return self.adata
         
   
