import os
import numpy as np
import anndata as ad
import pandas as pd

from scipy.sparse import csr_matrix

n_samples = 6
n_slices = [4, 7, 4, 5, 3, 4]

data_dir = '../data/3dst/data/'
save_dir = data_dir + 'preprocessed/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(n_samples):

    print(f'Sample RA{i+1}')

    if not os.path.exists(save_dir + f'RA{i+1}/'):
        os.makedirs(save_dir + f'RA{i+1}/')

    for j in range(n_slices[i]):

        print(f'Slice {j+1}')

        obs_name_list = []
        x_list = []
        y_list = []
        annotation_list = []

        with open(data_dir + f'RA{i+1}_{j+1}_annotations.txt', 'r') as file:
            next(file)
            for line in file:
                columns = line.split()
                if i == 5:
                    x_list.append(round(float(columns[2])))
                    y_list.append(round(float(columns[3])))
                    obs_name_list.append(f'{x_list[-1]}_{y_list[-1]}')
                else:
                    x_list.append(int(columns[2]))
                    y_list.append(int(columns[3]))
                    obs_name_list.append(columns[1])
                annotation_list.append(columns[4])

        n = len(obs_name_list)

        data_mtx = pd.read_csv(data_dir + f'RA{i+1}_Raw.exp_{j+1}.csv')
        if i != 5:
            data_mtx = data_mtx.set_index(data_mtx.columns[0])
        data_mtx = data_mtx.T
        data_mtx.columns.name = None

        m = len(list(data_mtx.index))

        p = 0
        q = 0

        filtered_obs_name_list = []
        filtered_x_list = []
        filtered_y_list = []
        filtered_annotation_list = []
        for k in range(len(list(data_mtx.index))):
            if list(data_mtx.index)[k] not in obs_name_list:
                q += 1  # spot without annotation
        for k in range(len(obs_name_list)):
            if obs_name_list[k] not in list(data_mtx.index):
                # print(obs_name_list[k])
                p += 1  # spot not in data matrix
            else:
                filtered_obs_name_list.append(obs_name_list[k])
                filtered_x_list.append(x_list[k])
                filtered_y_list.append(y_list[k])
                filtered_annotation_list.append(annotation_list[k])

        coordinates = np.array(list(zip(filtered_x_list, filtered_y_list)))

        print(list(set(annotation_list)))

        data_mtx = data_mtx.reindex(filtered_obs_name_list)

        adata = ad.AnnData(csr_matrix(data_mtx))
        adata.obs_names = filtered_obs_name_list
        adata.var_names = list(data_mtx.columns)
        adata.obs['annotations'] = filtered_annotation_list
        adata.obsm['spatial'] = coordinates

        print(n, m, p, q)
        print(n - p, m - q)
        if n - p != adata.shape[0] or m - q != adata.shape[0]:
            raise ValueError('Error !')
        print(adata.shape)

        adata.write_h5ad(save_dir + f"RA{i+1}/RA{i+1}_slice{j+1}.h5ad")

