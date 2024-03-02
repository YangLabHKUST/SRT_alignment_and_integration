import os
import csv
import anndata as ad

import STitch3D

import warnings
warnings.filterwarnings("ignore")

# RA
data_dir = '../../data/3dst/data/preprocessed/'
n_samples = 6
n_slices = [4, 7, 4, 5, 3, 4]

save_dir = '../../results/RA/'
if not os.path.exists(save_dir + 'STitch3D/'):
    os.makedirs(save_dir + 'STitch3D/')

pre_align_methods = ['PASTE_alignment', 'PASTE2']

for method in pre_align_methods:

    print(f'Using {method} as the pre-alignment method ...')

    for j in range(n_samples):

        print(f'RA{j+1}')

        slice_index_list = list(range(n_slices[j]))

        adata_st_list = []
        adata_st_list_raw = []

        for i in range(n_slices[j]):
            slice_raw = ad.read_h5ad(data_dir + f"RA{j+1}/RA{j+1}_slice{i+1}.h5ad")
            adata_st_list_raw.append(slice_raw)

            # load slices aligned by paste_alignment
            slice_aligned = ad.read_h5ad(save_dir + f'{method}/RA{j+1}_slice{i+1}_aligned.h5ad')
            slice_aligned.obsm['spatial_aligned'] = slice_aligned.obsm['spatial']
            adata_st_list.append(slice_aligned)

        adata_ref = ad.read_h5ad('../../data/scRNAdata_RA_Zhang2019/preprocessed/scRNAdata_RA_Zhang2019.h5ad')

        celltype_list_use = list(set(list(adata_ref.obs['celltype'])))
        # print(celltype_list_use)

        if j == 0:
            slice_dist = [21 for i in range(n_slices[j] - 1)]
        else:
            slice_dist = [7 for i in range(n_slices[j] - 1)]

        adata_st, adata_basis = STitch3D.utils.preprocess(adata_st_list,
                                                          adata_ref,
                                                          celltype_ref=celltype_list_use,
                                                          slice_dist_micron=slice_dist,
                                                          n_hvg_group=500,
                                                          c2c_dist=200)
        # print(adata_st)

        model = STitch3D.model.Model(adata_st, adata_basis, seed=1234)

        model.train()

        result = model.eval(adata_st_list_raw, save=False)

        with open(save_dir + f'STitch3D/{method}+STitch3D_embed_RA{j+1}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(model.adata_st.obsm['latent'])



