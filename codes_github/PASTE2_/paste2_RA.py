import os
import time
from paste2.PASTE2 import *
from paste2.model_selection import *
from paste2.projection import *
from paste2.helper import *

# RA
data_dir = '../../data/3dst/data/preprocessed/'
n_samples = 6
n_slices = [4, 7, 4, 5, 3, 4]

save_dir = '../../results/RA/'
if not os.path.exists(save_dir + 'PASTE2/'):
    os.makedirs(save_dir + 'PASTE2/')

for j in range(n_samples):

    print(f'RA {j+1}')

    slices = [ad.read_h5ad(data_dir + f"RA{j+1}/RA{j+1}_slice{i+1}.h5ad") for i in range(n_slices[j])]

    # Pairwise align the slices
    start = time.time()
    # calculate coverages
    ss = []
    for i in range(len(slices) - 1):
        s = select_overlap_fraction(slices[i], slices[i+1])
        ss.append(s)

    pis = []
    for i in range(len(slices) - 1):
        pi0 = match_spots_using_spatial_heuristic(slices[i].obsm['spatial'], slices[i+1].obsm['spatial'], use_ot=True)
        pi = partial_pairwise_align(slices[i], slices[i+1], min(ss[i], 0.99), G_init=pi0)
        pis.append(pi)
    print('Alignment Runtime: ' + str(time.time() - start))

    # To visualize the alignment you can stack the slices
    # according to the alignment pi
    new_slices = partial_stack_slices_pairwise(slices, pis)

    for i in range(len(new_slices)):
        new_slices[i].write_h5ad(save_dir + f'PASTE2/RA{j+1}_slice{i+1}_aligned.h5ad')
