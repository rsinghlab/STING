# STING - Spatial Transcriptomics cluster Inference with Nested GNNs

STING incorporates both graphs generated from the spatial proximity of tissue locations (or spots) and spot-specific graphs for related genes. This feature allows STING to better distinguish between clusters and identify meaningful gene-gene relations for knowledge discovery. It is a nested GNN framework that simultaneously models gene-gene and spatial relations. Using the gene expression, we generate a spot-specific gene-gene co-expression graph. We implement an inner GNN for these graphs to generate embeddings for each location. Next, we utilize these embeddings as features in a sample-wide graph generated using spatial information. We implement an outer GNN for this graph to reconstruct the original gene expression data. Finally, STING is trained end-to-end to generate embeddings that capture gene-gene and spatial information, which we input to a clustering algorithm to produce the spatial clusters. Experiments for 45 samples across 9 datasets and 5 spatial sequencing technologies show that STING outperforms the existing state-of-the-art techniques.

![STING Framework Overview](https://github.com/rsinghlab/STING/blob/main/STING%20Framework%20w%20Spot%20Groups.png?raw=true)

## Requirements
The model requires the following modules and versions - 

PyTorch >= 2.0.1

PyTorch Geometric >= 2.3.1

scanpy >= 1.9.1

Numpy < 2.0.0

Python Optimal Transport >= 0.9.1

Scikit-learn >= 1.2.2


We suggest generating an environment (such as conda) to run the code. You can use the given requirements.txt file to create a new environment. If it does not work, you can create the required conda environment directly by running the following lines sequentially in the shell.
```
conda create --name <env_name> python==3.11
conda activate <env_name>
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install scanpy
pip install POT
pip install "numpy<2"
```

## Downloading and Using Datasets
We use .h5ad files for our model. Annotated datasets used in the paper are available for download [here](https://cellxgene.cziscience.com/collections/0cca8620-8dee-45d0-aef5-23f032a5cf09) (MERMB), [here](https://db.cngb.org/stomics/datasets/STDS0000223/summary) (10xHK), and [here](http://sdmbench.drai.cn) (all others). The datasets, their ID and other details as displayed in the manuscript are tabulated below.

| Dataset ID | Technology  | Sample Count | Cluster Count | Spot Count     | Gene Count | Species | Tissue                |
|------------|-------------|--------------|----------------|----------------|------------|---------|------------------------|
| 10xHK      | 10x Visium  | 4            | 4, 5          | 1,949 - 4,860  | 9,943      | Human   | Kidney                |
| 10xHB      | 10x Visium  | 1            | 20            | 3,798          | 36,601     | Human   | Breast                |
| 10xHPC     | 10x Visium  | 12           | 5, 7          | 3,460 - 4,789  | 33,538     | Human   | Prefrontal Cortex     |
| STAR1      | STARmap     | 3            | 4             | 1,049 - 1,088  | 166        | Mouse   | Prefrontal Cortex     |
| STAR2      | STARmap     | 1            | 7             | 1,207          | 1,020      | Mouse   | Prefrontal Cortex     |
| BAR        | BaristaSeq  | 3            | 6             | 1,525 - 2,042  | 79         | Mouse   | Primary Cortex        |
| OSM        | osmFISH     | 1            | 11            | 4,839          | 33         | Mouse   | Somatosensory Cortex  |
| MERMPH     | MERFISH     | 5            | 8             | 5,488 - 5,926  | 155        | Mouse   | Preoptic Hypothalamus |
| MERMB      | MERFISH     | 15 (chosen from a total 245 samples)          | 5 - 12        | 6,000          | 1,122      | Mouse   | Brain                 |

<br>
To use your own spatial transcriptomics file, use anndata to read the file and store it in an anndata object. The anndata object (named `adata` in the tutorial) should be formatted with `adata.X` containing the raw counts and `adata.obsm['spatial']` containing the spatial locations of the spots.

## Tutorial
To see how to use STING to generate embeddings and attention scores, please refer to tutorial.ipynb.

## Bugs and Suggestions
Please report any bugs, problems, suggestions, or requests as a Github issue.
