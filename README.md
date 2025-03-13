# STING - Spatial Transcriptomics cluster Inference with Nested GNNs

STING incorporates both graphs generated from the spatial proximity of tissue locations (or spots) and spot-specific graphs for related genes. This feature allows STING to better distinguish between clusters and identify meaningful gene-gene relations for knowledge discovery. It is a nested GNN framework that simultaneously models gene-gene and spatial relations. Using the gene expression, we generate a spot-specific gene-gene co-expression graph. We implement an inner GNN for these graphs to generate embeddings for each location. Next, we utilize these embeddings as features in a sample-wide graph generated using spatial information. We implement an outer GNN for this graph to reconstruct the original gene expression data. Finally, STING is trained end-to-end to generate embeddings that capture gene-gene and spatial information, which we input to a clustering algorithm to produce the spatial clusters. Experiments for 26 samples across 7 datasets and 5 spatial sequencing technologies show that STING outperforms the existing state-of-the-art techniques.

[Preprint](https://www.biorxiv.org/content/10.1101/2025.02.03.636316v1.abstract) is available.

![STING Framework Overview](https://github.com/rsinghlab/STING/blob/main/STING%20Framework.png?raw=true)

## Requirements
We suggest generating an environment (such as conda) to run the code. We have provided the requirements in requirements.txt.

You can create a conda environment directly by running this line in the shell.
```
conda create --name <env_name> --file requirements.txt
```

## Downloading Datasets
We use .h5ad files for our model. Annotated datasets used in the paper are available for download [here](sdmbench.drai.cn).
