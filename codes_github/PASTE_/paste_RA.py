import os
import time
import csv
import ot
import torch
import anndata as ad
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
if not os.path.exists(save_dir + 'PASTE_alignment/'):
    os.makedirs(save_dir + 'PASTE_alignment/')
if not os.path.exists(save_dir + 'PASTE_integration/'):
    os.makedirs(save_dir + 'PASTE_integration/')


import paste as pst

for j in range(n_samples):

    print(f'RA {j+1}')

    slices = [ad.read_h5ad(data_dir + f"RA{j+1}/RA{j+1}_slice{i+1}.h5ad") for i in range(n_slices[j])]

    # Pairwise align the slices
    start = time.time()
    pis = []
    for i in range(len(slices) - 1):
        pi0 = pst.match_spots_using_spatial_heuristic(slices[i].obsm['spatial'], slices[i+1].obsm['spatial'], use_ot=True)
        pi = pst.pairwise_align(slices[i], slices[i+1], G_init=pi0, norm=True, verbose=False,
                                backend=ot.backend.TorchBackend(), use_gpu=True)
        pis.append(pi)
    print('Alignment Runtime: ' + str(time.time() - start))

    # To visualize the alignment you can stack the slices
    # according to the alignment pi
    new_slices = pst.stack_slices_pairwise(slices, pis)

    for i in range(len(new_slices)):
        new_slices[i].write_h5ad(save_dir + f'PASTE_alignment/RA{j+1}_slice{i+1}_aligned.h5ad')

# paste_integration
for j in range(n_samples):

    print(f'RA {j+1}')

    slices = [ad.read_h5ad(data_dir + f"RA{j+1}/RA{j+1}_slice{i+1}.h5ad") for i in range(n_slices[j])]

    initial_slice = slices[1].copy()
    lmbda = len(slices) * [1 / len(slices)]  # set hyperparameter to be uniform

    # Possible to pass in an initial pi (as keyword argument pis_init)
    # to improve performance, see Tutorial.ipynb notebook for more details.
    init_pis = [pst.match_spots_using_spatial_heuristic(np.array(initial_slice.obsm['spatial']),
                                                        slices[i].obsm['spatial'], use_ot=True) for i in range(len(slices))]
    start = time.time()
    center_slice, pis = pst.center_align(initial_slice, slices, lmbda, random_seed=5, norm=True, verbose=True,
                                         pis_init=init_pis, backend=ot.backend.TorchBackend(), use_gpu=True)
    print('Integration Runtime: ' + str(time.time() - start))

    with open(save_dir + f'PASTE_integration/PASTE_integration_embed_RA{j+1}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(center_slice.uns['paste_W'])

