import matplotlib.patches as mpatches
import time
from ..src.paste2.PASTE2 import *
from ..src.paste2.model_selection import *
from ..src.paste2.projection import *
from ..src.paste2.helper import *
from ..utils import *


sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
adatas = {sample:sc.read_h5ad('../../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
sample_groups = [["151507", "151508", "151509","151510"], [ "151669", "151670","151671", "151672"], [ "151673","151674", "151675", "151676"]]
for i in range(len(sample_groups)):
    for j in range(len(sample_groups[i])):
        if j % 2 == 0:
            adatas[sample_groups[i][j]] = partial_cut(adatas[sample_groups[i][j]], 0.85, is_left=True)
        else:
            adatas[sample_groups[i][j]] = partial_cut(adatas[sample_groups[i][j]], 0.85, is_left=False)
layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]
layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]

slice_map = {0:'A',1:'B',2:'C',3:'D'}
sample_map = {0:'A',1:'B',2:'C'}


for sample_choose in range(len(sample_groups)):
    slice1 = adatas[sample_groups[sample_choose][0]]
    slice2 = adatas[sample_groups[sample_choose][1]]
    slice3 = adatas[sample_groups[sample_choose][2]]
    slice4 = adatas[sample_groups[sample_choose][3]]

    # Pairwise align the slices
    start = time.time()
    s1 = select_overlap_fraction(slice1, slice2)
    s2 = select_overlap_fraction(slice2, slice3)
    s3 = select_overlap_fraction(slice3, slice4)
    ss = [s1, s2, s3]

    pi0 = match_spots_using_spatial_heuristic(slice1.obsm['spatial'],slice2.obsm['spatial'], use_ot=True)
    pi12 = partial_pairwise_align(slice1, slice2, s1, G_init=pi0)
    pi0 = match_spots_using_spatial_heuristic(slice2.obsm['spatial'], slice3.obsm['spatial'], use_ot=True)
    pi23 = partial_pairwise_align(slice2, slice3, s2, G_init=pi0)
    pi0 = match_spots_using_spatial_heuristic(slice3.obsm['spatial'], slice4.obsm['spatial'], use_ot=True)
    pi34 = partial_pairwise_align(slice3, slice4, s3, G_init=pi0)
    print('Alignment Runtime: ' + str(time.time() - start))

    # To visualize the alignment you can stack the slices
    # according to the alignment pi
    slices, pis = [slice1, slice2, slice3, slice4], [pi12, pi23, pi34]
    new_slices = partial_stack_slices_pairwise(slices, pis)

    # pis_dict = {}
    # for i, matrix in enumerate(pis):
    #     pis_dict[f'pi_{i}'] = matrix
    # pis_dict['ss'] = ss
    # np.savez('../../results/partial_DLPFC_0.85/paste2_pis_Sample{}_partial_DLPFC.npz'.format(sample_map[sample_choose]), **pis_dict)
    #
    # pis_dict = np.load('../../results/partial_DLPFC_0.85/paste2_pis_Sample{}_partial_DLPFC.npz'.format(sample_map[sample_choose]))
    # print(sample_map[sample_choose])
    # acc12 = mapping_accuracy(layer_groups[sample_choose][0].obs['layer_guess_reordered'],
    #                          layer_groups[sample_choose][1].obs['layer_guess_reordered'], pis_dict['pi_0'])
    # acc23 = mapping_accuracy(layer_groups[sample_choose][1].obs['layer_guess_reordered'],
    #                          layer_groups[sample_choose][2].obs['layer_guess_reordered'], pis_dict['pi_1'])
    # acc34 = mapping_accuracy(layer_groups[sample_choose][2].obs['layer_guess_reordered'],
    #                          layer_groups[sample_choose][3].obs['layer_guess_reordered'], pis_dict['pi_2'])
    # print(pis_dict['ss'])
    # print(acc12 / pis_dict['ss'][0], acc23 / pis_dict['ss'][1], acc34 / pis_dict['ss'][2])
    #
    # pis_dict = np.load('../../results/partial_DLPFC/paste2_pis_Sample{}_partial_DLPFC.npz'.format(sample_map[sample_choose]))
    # slices, pis = [slice1, slice2, slice3, slice4], [pis_dict['pi_0'], pis_dict['pi_1'], pis_dict['pi_2']]
    # new_slices = partial_stack_slices_pairwise(slices, pis)
    # # print(new_slices)
    # for i, anndata_obj in enumerate(new_slices):
    #     file_name = "../../results/stitch3d_use/partial_DLPFC_0.7/paste2_Sample{}_slice{}_partial_DLPFC.h5ad".format(
    #         sample_map[sample_choose], slice_map[i])
    #     anndata_obj.write(file_name)

    plt.figure(figsize=(7, 7))
    for i in range(len(layer_groups[sample_choose])):
        adata = new_slices[i]
        colors = list(adata.obs['layer_guess_reordered'].astype('str').map(layer_to_color_map))
        plt.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], linewidth=0, s=70, marker=".",
                    color=colors)
    plt.title('Sample' + sample_map[sample_choose], size=12)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.legend(handles=[mpatches.Patch(color=layer_to_color_map[adata.obs['layer_guess_reordered'].cat.categories[i]],
                                       label=adata.obs['layer_guess_reordered'].cat.categories[i]) for i in
                        range(len(adata.obs['layer_guess_reordered'].cat.categories))], fontsize=6,
               title='Cortex layer', title_fontsize=6, bbox_to_anchor=(1, 1))
    save_path = "../../results/partial_DLPFC_0.85/paste2_Sample{}_partial_DLPFC.png".format(sample_map[sample_choose])
    plt.savefig(save_path)

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    for vv in range(3):
        adata = new_slices[vv]
        adata2 = new_slices[vv + 1]
        colors = list(adata.obs['layer_guess_reordered'].astype('str').map(layer_to_color_map))
        colors2 = list(adata2.obs['layer_guess_reordered'].astype('str').map(layer_to_color_map))
        axs[int(vv / 2), int(vv % 2)].scatter(
            adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1],
            linewidth=0, s=20, alpha=1, marker=".", color=colors
        )
        axs[int(vv / 2), int(vv % 2)].scatter(
            adata2.obsm['spatial'][:, 0], adata2.obsm['spatial'][:, 1],
            linewidth=0, s=20, alpha=1, marker=".", color=colors2
        )
        axs[int(vv / 2), int(vv % 2)].axis('off')
        axs[int(vv / 2), int(vv % 2)].invert_yaxis()
    fig.delaxes(axs[1, 1])
    save_path = "../../results/partial_DLPFC_0.85/paste2_individual_Sample{}_partial_DLPFC.png".format(
        sample_map[sample_choose])
    plt.savefig(save_path)
plt.show()
