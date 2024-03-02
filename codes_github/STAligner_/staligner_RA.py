import os
import csv
import torch
import scipy
import anndata as ad
import scanpy as sc
import numpy as np

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# RA
data_dir = '../../data/3dst/data/preprocessed/'
n_samples = 6
n_slices = [4, 7, 4, 5, 3, 4]

save_dir = '../../results/RA/'
if not os.path.exists(save_dir + 'STAligner/'):
    os.makedirs(save_dir + 'STAligner/')

import STAligner

# sample specific
for j in range(n_samples):

    print(f'RA {j+1}')

    slice_index_list = list(range(n_slices[j]))

    cas_list = []
    adj_list = []

    for i in range(n_slices[j]):

        print(i)

        adata = ad.read_h5ad(data_dir + f"RA{j+1}/RA{j+1}_slice{i+1}.h5ad")

        adata.obs_names = [x + f'_RA{j}_slice{i}' for x in adata.obs_names]

        # Constructing the spatial network
        if j < 2:
            STAligner.Cal_Spatial_Net(adata, rad_cutoff=1.5)  # the spatial network are saved in adata.uns[‘adj’]
        else:
            STAligner.Cal_Spatial_Net(adata, rad_cutoff=2.5)  # the spatial network are saved in adata.uns[‘adj’]

        # Normalization
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
        adata = adata[:, adata.var['highly_variable']]

        adj_list.append(adata.uns['adj'])
        cas_list.append(adata)

    adata_concat = ad.concat(cas_list, label="slice_index", keys=slice_index_list)
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_index"].astype(str).astype('category')
    print('adata_concat.shape: ', adata_concat.shape)

    adj_concat = np.asarray(adj_list[0].todense())
    for batch_id in range(1, len(slice_index_list)):
        adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
    adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

    adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=50,
                                             device=device, random_seed=1234)

    with open(save_dir + f'STAligner/STAligner_embed_RA{j+1}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adata_concat.obsm['STAligner'])


# joint samples
class_list = ['Seropositive', 'Seronegative']
slice_name_list = []
for i in range(2):
    name_list = []
    if i == 0:
        for j in range(3):
            for k in range(n_slices[j]):
                name_list.append(f"RA{j+1}/RA{j+1}_slice{k+1}.h5ad")
    else:
        for j in range(3, 6):
            for k in range(n_slices[j]):
                name_list.append(f"RA{j+1}/RA{j+1}_slice{k+1}.h5ad")
    slice_name_list.append(name_list)

for j in range(len(class_list)):

    print(class_list[j])

    cas_list = []
    adj_list = []

    for i, slice_name in enumerate(slice_name_list[j]):

        print(slice_name)

        adata = ad.read_h5ad(data_dir + slice_name)

        adata.obs_names = [x + slice_name for x in adata.obs_names]

        # Constructing the spatial network
        if j == 0 and i < 11:
            STAligner.Cal_Spatial_Net(adata, rad_cutoff=1.5)  # the spatial network are saved in adata.uns[‘adj’]
        else:
            STAligner.Cal_Spatial_Net(adata, rad_cutoff=2.5)  # the spatial network are saved in adata.uns[‘adj’]

        # Normalization
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
        adata = adata[:, adata.var['highly_variable']]

        adj_list.append(adata.uns['adj'])
        cas_list.append(adata)

    adata_concat = ad.concat(cas_list, label="slice_name", keys=slice_name_list[j])
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype(str).astype('category')
    print('adata_concat.shape: ', adata_concat.shape)

    adj_concat = np.asarray(adj_list[0].todense())
    for batch_id in range(1, len(slice_name_list[j])):
        adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
    adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

    adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=50,
                                             device=device, random_seed=1234)

    with open(save_dir + f'STAligner/STAligner_embed_{class_list[j]}_RA.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adata_concat.obsm['STAligner'])
