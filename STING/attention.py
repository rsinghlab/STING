import numpy as np


def get_attention_matrices(adata, clustered = True):
    '''
    Input parameters:
    adata - An anndata object. Adata.var.index should contain the gene names and adata.uns.['a1'] should have the attention scores from running the STING model.
    clustered - boolean value. If True, the function returns the attention scores per cluster. In this case, adata.obs['clusters'] should contain the clusters. If False, the function returns the attention scores for the entire cluster.
    
    Outputs:
    final_avgs - a (c*)g*g sized numpy array, where g = number of highly variable genes. If clustered = True, c = number of clusters. If clustered = False, the output is of size g*g.
    hvg - a numpy array with the order of the highly variable genes in the final_avgs attention matrix. 
    (Optional) clusters - a numpy array returned when clustered = True. It contains the order of the clusters in final_avg. For example, if "Cluster1" is in the 0-th position in clusters, then final_avgs[0] contains the attention scores of "Cluster1". 
    '''

    adata_Vars =  adata[:, adata.var['highly_variable']]
    adata.uns['HVG_list'] = list(adata_Vars.var.index)

    hvg = adata.uns['HVG_list']
    gcount = len(hvg)
    

    
    
    if clustered:
        clusters = np.array(adata.obs['clusters'])
        un = np.unique(clusters)
        classes = {}
        for i in range(len(un)):
            classes[un[i]] = i

        num_classes = len(list(classes.keys()))
        
        counts = np.zeros((num_classes,gcount,gcount))
        avgs = np.zeros((num_classes,gcount,gcount))
        

        attention_no = 'a1'
        for edge in range(adata.uns[attention_no][0].shape[1]):
            node1 = int(adata.uns[attention_no][0][0][edge])
            node2 = int(adata.uns[attention_no][0][1][edge])
            cell_no = node1//gcount
            gene1 = node1%gcount
            gene2 = node2%gcount
            cls_no = classes[clusters[cell_no]]

            counts[cls_no,gene1, gene2] += 1
            avgs[cls_no, gene1, gene2] += float(adata.uns[attention_no][1][edge])
            
        final_avgs = np.nan_to_num(np.divide(avgs + np.transpose(avgs, (0,2,1)), counts + np.transpose(counts, (0,2,1))))
        return final_avgs, hvg, clusters
            
    else:
        counts = np.zeros((gcount,gcount))
        avgs = np.zeros((gcount,gcount))

        attention_no = 'a1'
        for edge in range(adata.uns[attention_no][0].shape[1]):
            node1 = int(adata.uns[attention_no][0][0][edge])
            node2 = int(adata.uns[attention_no][0][1][edge])
            cell_no = node1//gcount
            gene1 = node1%gcount
            gene2 = node2%gcount

            counts[gene1, gene2] += 1
            avgs[gene1, gene2] += float(adata.uns[attention_no][1][edge])
            
        final_avgs = np.nan_to_num(np.divide(avgs + avgs.T, counts + counts.T))
        return final_avgs, hvg