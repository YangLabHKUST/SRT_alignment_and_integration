from utils import *
import seaborn as sns
import paste as pst

from paste2.helper import *
from paste2.projection import *


slice_map = {0:'A',1:'B',2:'C',3:'D'}
sample_map = {0:'A',1:'B',2:'C'}

# load DLPFC dataset
# using DLPFC as an example, the results of other datasets can be measured as the same way
sample_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674","151675", "151676"]
adatas = {sample: sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
sample_groups = [["151507", "151508", "151509", "151510"], ["151669", "151670", "151671", "151672"],["151673", "151674", "151675", "151676"]]
layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]
layer_to_color_map = {'Layer{0}'.format(i + 1): sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]

for i in range(len(sample_groups)):
    print('Sample ' + sample_map[i])
    iter_comb = [(1, 0), (2, 1), (3, 2)]

    all_ratio = []
    all_accu = []
    all_mean_accu = []


    print('raw')
    layer_group = layer_groups[i].copy()
    # rename the column 'layer_guess_reordered' to 'Ground Truth' for a consensus
    for slice in layer_group:
        slice.obs.rename(columns={'layer_guess_reordered': 'Ground Truth'}, inplace=True)
    adata = ad.concat(layer_group, label="slice_name", keys=sample_groups[i])
    adata.obs['Ground Truth'] = adata.obs['Ground Truth'].astype('category')
    adata.obs["batch_name"] = adata.obs["slice_name"].astype('category')
    ratios = []
    accus = []
    for comb in iter_comb:
        k, j = comb[0], comb[1]
        adata_target = layer_group[k]
        adata_ref = layer_group[j]
        slice_target = sample_groups[i][k]
        slice_ref = sample_groups[i][j]

        ratio, accuracy = ratio_and_accuracy(adata, adata_target, adata_ref, slice_target, slice_ref,
                                              use_rep='spatial', ratio=0.8)
        ratios.append(ratio)
        accus.append(accuracy)
    print('ratio:', ratios)
    print('accuracy:', accus)
    all_ratio.append(ratios)
    all_accu.append(accus)
    all_mean_accu.append(round(sum(accus) / len(accus), 2))


    print('PASTE')
    # load mapping results (pis)
    pis_dict = np.load('../results/DLPFC/paste_alignment_pis_Sample{}_DLPFC.npz'.format(sample_map[i]))
    slices, pis = layer_group, [pis_dict['pi_0'], pis_dict['pi_1'], pis_dict['pi_2']]
    new_slices = pst.stack_slices_pairwise(slices, pis)
    for slice in new_slices:
        slice.obs.rename(columns={'layer_guess_reordered': 'Ground Truth'}, inplace=True)
    adata = ad.concat(new_slices, label="slice_name", keys=sample_groups[i])
    adata.obs['Ground Truth'] = adata.obs['Ground Truth'].astype('category')
    adata.obs["batch_name"] = adata.obs["slice_name"].astype('category')
    ratios = []
    accus = []
    for comb in iter_comb:
        k, j = comb[0], comb[1]
        adata_target = new_slices[k]
        adata_ref = new_slices[j]
        slice_target = sample_groups[i][k]
        slice_ref = sample_groups[i][j]

        ratio, accuracy = ratio_and_accuracy(adata, adata_target, adata_ref, slice_target, slice_ref,
                                              use_rep='spatial', ratio=0.8)
        ratios.append(ratio)
        accus.append(accuracy)
    print('ratio:', ratios)
    print('accuracy:', accus)
    all_ratio.append(ratios)
    all_accu.append(accus)
    all_mean_accu.append(round(sum(accus) / len(accus), 2))


    print('PASTE2')
    # load mapping results (pis)
    pis_dict = np.load('../results/DLPFC/paste2_pis_Sample{}_DLPFC.npz'.format(sample_map[i]))
    slices, pis = layer_group, [pis_dict['pi_0'], pis_dict['pi_1'], pis_dict['pi_2']]
    new_slices = partial_stack_slices_pairwise(slices, pis)
    for slice in new_slices:
        slice.obs.rename(columns={'layer_guess_reordered': 'Ground Truth'}, inplace=True)
    adata = ad.concat(new_slices, label="slice_name", keys=sample_groups[i])
    adata.obs['Ground Truth'] = adata.obs['Ground Truth'].astype('category')
    adata.obs["batch_name"] = adata.obs["slice_name"].astype('category')
    ratios = []
    accus = []
    for comb in iter_comb:
        k, j = comb[0], comb[1]
        adata_target = new_slices[k]
        adata_ref = new_slices[j]
        slice_target = sample_groups[i][k]
        slice_ref = sample_groups[i][j]

        ratio, accuracy = ratio_and_accuracy(adata, adata_target, adata_ref, slice_target, slice_ref,
                                              use_rep='spatial', ratio=0.8)
        ratios.append(ratio)
        accus.append(accuracy)
    print('ratio:', ratios)
    print('accuracy:', accus)
    all_ratio.append(ratios)
    all_accu.append(accus)
    all_mean_accu.append(round(sum(accus) / len(accus), 2))


    print('GPSA')
    spots_count = [0]
    n = 0
    for j in range(len(layer_groups[i])):
        num = layer_groups[i][j].shape[0]
        n += num
        spots_count.append(n)
    view_idx = [np.arange(spots_count[ii], spots_count[ii + 1]) for ii in range(len(layer_groups[i]))]
    # load aligned coordinates
    coords_data = sc.read_csv("../results/DLPFC/gpsa_Sample{}_DLPFC_aligned_coords_st.csv".format(sample_map[i])).X[1:]

    layer_group = layer_groups[i].copy()
    for j in range(len(layer_group)):
        layer_group[j].obs.rename(columns={'layer_guess_reordered': 'Ground Truth'}, inplace=True)
        layer_group[j].obsm['spatial'] = coords_data[view_idx[j]]
    adata = ad.concat(layer_group, label="slice_name", keys=sample_groups[i])
    adata.obs['Ground Truth'] = adata.obs['Ground Truth'].astype('category')
    adata.obs["batch_name"] = adata.obs["slice_name"].astype('category')
    ratios = []
    accus = []
    for comb in iter_comb:
        k, j = comb[0], comb[1]
        adata_target = layer_group[k]
        adata_ref = layer_group[j]
        slice_target = sample_groups[i][k]
        slice_ref = sample_groups[i][j]

        ratio, accuracy = ratio_and_accuracy(adata, adata_target, adata_ref, slice_target, slice_ref,
                                              use_rep='spatial', ratio=0.8)
        ratios.append(ratio)
        accus.append(accuracy)
    print('ratio:', ratios)
    print('accuracy:', accus)
    all_ratio.append(ratios)
    all_accu.append(accus)
    all_mean_accu.append(round(sum(accus) / len(accus), 2))


    print('STAligner')
    layer_group = layer_groups[i].copy()
    n = 0
    for slice in layer_group:
        slice.obs.rename(columns={'layer_guess_reordered': 'Ground Truth'}, inplace=True)
        slice.obs_names = [x + '_' + sample_groups[i][n] for x in slice.obs_names]
        n += 1
    # load the joint data with latent representations
    adata = sc.read_h5ad('../results/DLPFC/staligner_Sample{}_DLPFC.h5ad'.format(sample_map[i]))
    # align the slices using ICP according to mnn pairs
    for comb in iter_comb:
        k, j = comb[0], comb[1]
        adata_target = layer_group[k]
        adata_ref = layer_group[j]
        slice_target = sample_groups[i][k]
        slice_ref = sample_groups[i][j]

        aligned_coor = ICP_align(adata, adata_target, adata_ref, slice_target, slice_ref)
        adata_target.obsm["spatial"] = aligned_coor
        layer_group[k].obsm["spatial"] = aligned_coor

    adata = ad.concat(layer_group, label="slice_name", keys=sample_groups[i])
    adata.obs['Ground Truth'] = adata.obs['Ground Truth'].astype('category')
    adata.obs["batch_name"] = adata.obs["slice_name"].astype('category')
    ratios = []
    accus = []
    for comb in iter_comb:
        k, j = comb[0], comb[1]
        adata_target = layer_group[k]
        adata_ref = layer_group[j]
        slice_target = sample_groups[i][k]
        slice_ref = sample_groups[i][j]

        ratio, accuracy = ratio_and_accuracy(adata, adata_target, adata_ref, slice_target, slice_ref,
                                              use_rep='spatial', ratio=0.8)
        ratios.append(ratio)
        accus.append(accuracy)
    print('ratio:', ratios)
    print('accuracy:', accus)
    all_ratio.append(ratios)
    all_accu.append(accus)
    all_mean_accu.append(round(sum(accus) / len(accus), 2))


    # save the results
    results_dict = {
        "all_ratio": all_ratio,
        "all_accu": all_accu,
        "all_mean_accu": all_mean_accu
    }
    np.savez('../results/accuracy/ratio_accuracy_Sample{}_DLPFC.npz'.format(sample_map[i]), **results_dict)

