import os
import anndata as ad
import pandas as pd

from scipy.sparse import csr_matrix

data_dir = '../data/scRNAdata_RA_Zhang2019/'
save_dir = data_dir + 'preprocessed/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# data_matrix = pd.read_csv(data_dir + 'celseq_matrix_ru10_molecules.tsv', sep='\t', header=0, index_col=0)
# data_matrix.fillna(0, inplace=True)
# data_matrix.index.name = None
# data_matrix = data_matrix.astype(int)
# data_matrix.to_csv(data_dir + 'celseq_matrix_ru10_molecules.csv')

data_matrix = pd.read_csv(data_dir + 'celseq_matrix_ru10_molecules.csv', index_col=0)
# print(data_matrix.shape)

cell_names = data_matrix.columns.tolist()
gene_names = data_matrix.index.tolist()

meta_data = pd.read_csv(data_dir + 'celseq_meta.tsv', sep='\t', header=0)

if cell_names == list(meta_data['cell_name']):
    print("Correct order")

adata = ad.AnnData(csr_matrix(data_matrix.T))
adata.obs_names = cell_names
adata.var_names = gene_names
adata.obs['celltype'] = list(meta_data['type'])
adata.obs['disease'] = list(meta_data['disease'])
# print(adata.shape)

adata = adata[adata.obs['disease'] == 'RA']
adata = adata[adata.obs['celltype'] != 'Empty']
# print(adata.shape)
adata.write_h5ad(save_dir + f"scRNAdata_RA_Zhang2019.h5ad")

# adata = ad.read_h5ad(save_dir + f"scRNAdata_RA_Zhang2019.h5ad")
# print(list(set(list(adata.obs['celltype']))))



