import matplotlib.patches as mpatches
import paste as pst
import time
from codes_github.utils import *


# load data
sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
adatas = {sample:sc.read_h5ad('../../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
sample_groups = [["151507", "151508", "151509","151510"], [ "151669", "151670","151671", "151672"], [ "151673","151674", "151675", "151676"]]
layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]
layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]

slice_map = {0:'A',1:'B',2:'C',3:'D'}
sample_map = {0:'A',1:'B',2:'C'}

# plot raw data
fig, axs = plt.subplots(3, 4,figsize=(15,11.5))
for j in range(len(layer_groups)):
    axs[j,0].text(-0.1, 0.5, 'Sample '+slice_map[j],fontsize=12,rotation='vertical',transform = axs[j,0].transAxes,verticalalignment='center')
    for i in range(len(layer_groups[j])):
        adata = adatas[sample_list[j*4+i]]
        colors = list(adata.obs['layer_guess_reordered'].astype('str').map(layer_to_color_map))
        axs[j,i].scatter(layer_groups[j][i].obsm['spatial'][:,0],layer_groups[j][i].obsm['spatial'][:,1],linewidth=0,s=20, marker=".",color=colors)
        axs[j,i].set_title('Slice '+ slice_map[i],size=12)
        axs[j,i].invert_yaxis()
        axs[j,i].axis('off')
        if i<3:
            s = '300$\mu$m' if i==1 else "10$\mu$m"
            delta = 0.05 if i==1 else 0
            axs[j,i].annotate('',xy=(1-delta, 0.5), xytext=(1.2+delta, 0.5),xycoords=axs[j,i].transAxes,textcoords=axs[j,i].transAxes,arrowprops=dict(arrowstyle='<->',lw=1))
            axs[j,0].text(1.1, 0.55, s,fontsize=9,transform = axs[j,i].transAxes,horizontalalignment='center')
    axs[j,3].legend(handles=[mpatches.Patch(color=layer_to_color_map[adata.obs['layer_guess_reordered'].cat.categories[i]], label=adata.obs['layer_guess_reordered'].cat.categories[i]) for i in range(len(adata.obs['layer_guess_reordered'].cat.categories))],fontsize=10,title='Cortex layer',title_fontsize=12,bbox_to_anchor=(1, 1))
save_path = "../../results/DLPFC/DLPFC.png"
plt.savefig(save_path)
plt.show()


for sample_choose in range(len(sample_groups)):
    slice1 = adatas[sample_groups[sample_choose][0]]
    slice2 = adatas[sample_groups[sample_choose][1]]
    slice3 = adatas[sample_groups[sample_choose][2]]
    slice4 = adatas[sample_groups[sample_choose][3]]
    slices = [slice1, slice2, slice3, slice4]

    # plot raw sample
    plt.figure(figsize=(7, 7))
    for i in range(len(layer_groups[sample_choose])):
        adata = slices[i]
        colors = list(adata.obs['layer_guess_reordered'].astype('str').map(layer_to_color_map))
        plt.scatter(layer_groups[sample_choose][i].obsm['spatial'][:, 0],
                    layer_groups[sample_choose][i].obsm['spatial'][:, 1], linewidth=0, s=70, marker=".", color=colors)
    plt.title('Sample' + sample_map[sample_choose], size=12)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.legend(handles=[mpatches.Patch(color=layer_to_color_map[adata.obs['layer_guess_reordered'].cat.categories[i]],
                                       label=adata.obs['layer_guess_reordered'].cat.categories[i]) for i in
                        range(len(adata.obs['layer_guess_reordered'].cat.categories))], fontsize=6,
               title='Cortex layer', title_fontsize=6, bbox_to_anchor=(1, 1))
    save_path = "../../results/DLPFC/paste_raw_Sample{}_DLPFC.png".format(sample_map[sample_choose])
    plt.savefig(save_path)
    plt.show()

    # Pairwise align the slices
    start = time.time()
    pi0 = pst.match_spots_using_spatial_heuristic(slice1.obsm['spatial'],slice2.obsm['spatial'], use_ot=True)
    pi12 = pst.pairwise_align(slice1, slice2, G_init=pi0,norm=True,verbose=False,
                              backend=ot.backend.TorchBackend(), use_gpu=True)
    pi0 = pst.match_spots_using_spatial_heuristic(slice2.obsm['spatial'], slice3.obsm['spatial'], use_ot=True)
    pi23 = pst.pairwise_align(slice2, slice3, G_init=pi0,norm=True,verbose=False,
                              backend=ot.backend.TorchBackend(), use_gpu=True)
    pi0 = pst.match_spots_using_spatial_heuristic(slice3.obsm['spatial'], slice4.obsm['spatial'], use_ot=True)
    pi34 = pst.pairwise_align(slice3, slice4, G_init=pi0,norm=True,verbose=False,
                              backend=ot.backend.TorchBackend(), use_gpu=True)
    print('Alignment Runtime: ' + str(time.time() - start))

    # To visualize the alignment you can stack the slices
    # according to the alignment pi
    slices, pis = [slice1, slice2, slice3, slice4], [pi12, pi23, pi34]
    new_slices = pst.stack_slices_pairwise(slices, pis)

    # # save results
    # pis_dict = {}
    # for i, matrix in enumerate(pis):
    #     pis_dict[f'pi_{i}'] = matrix
    # np.savez('../../results/DLPFC/paste_alignment_pis_Sample{}_DLPFC.npz'.format(sample_map[sample_choose]), **pis_dict)
    # pis_dict = np.load('../results/DLPFC/paste_alignment_pis_Sample{}_DLPFC.npz'.format(sample_map[sample_choose]))
    # print(sample_map[sample_choose])

    # # mapping accuracy
    # acc12 = mapping_accuracy(layer_groups[sample_choose][0].obs['layer_guess_reordered'],
    #                          layer_groups[sample_choose][1].obs['layer_guess_reordered'], pis_dict['pi_0'])
    # acc23 = mapping_accuracy(layer_groups[sample_choose][1].obs['layer_guess_reordered'],
    #                          layer_groups[sample_choose][2].obs['layer_guess_reordered'], pis_dict['pi_1'])
    # acc34 = mapping_accuracy(layer_groups[sample_choose][2].obs['layer_guess_reordered'],
    #                          layer_groups[sample_choose][3].obs['layer_guess_reordered'], pis_dict['pi_2'])
    # print(acc12, acc23, acc34)

    # # load results
    # pis_dict = np.load('../../results/DLPFC/paste_alignment_pis_Sample{}_DLPFC.npz'.format(sample_map[sample_choose]))
    # slices, pis = [slice1, slice2, slice3, slice4], [pis_dict['pi_0'], pis_dict['pi_1'], pis_dict['pi_2']]
    # new_slices = pst.stack_slices_pairwise(slices, pis)
    #
    # # save data with new coordinates, which is used for STitch3D model
    # for i, anndata_obj in enumerate(new_slices):
    #     file_name = "../../results/stitch3d_use/DLPFC/paste_alignment_Sample{}_slice{}_DLPFC.h5ad".format(sample_map[sample_choose], slice_map[i])
    #     anndata_obj.write(file_name)

    # plot paste_alignment results
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
    save_path = "../../results/DLPFC/paste_alignment_Sample{}_DLPFC.png".format(sample_map[sample_choose])
    plt.savefig(save_path)
    # plt.show()

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
    save_path = "../../results/DLPFC/paste_individual_Sample{}_DLPFC.png".format(
        sample_map[sample_choose])
    plt.savefig(save_path)
    # plt.show()

plt.show()

# paste_integration
# We have to reload the slices as pairwise_alignment modifies the slices.
sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
adatas = {sample:sc.read_h5ad('../../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
sample_groups = [["151507", "151508", "151509","151510"], [ "151669", "151670","151671", "151672"], [ "151673","151674", "151675", "151676"]]
layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]
layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]

slice_map = {0:'A',1:'B',2:'C',3:'D'}
sample_map = {0:'A',1:'B',2:'C'}

# Construct a center slice
# choose one of the slices as the coordinate reference for the center slice,
# i.e. the center slice will have the same number of spots as this slice and
# the same coordinates.
for sample_choose in range(len(sample_groups)):
    slice1 = adatas[sample_groups[sample_choose][0]]
    slice2 = adatas[sample_groups[sample_choose][1]]
    slice3 = adatas[sample_groups[sample_choose][2]]
    slice4 = adatas[sample_groups[sample_choose][3]]
    slices = [slice1, slice2, slice3, slice4]
    initial_slice = slice2.copy()
    lmbda = len(slices) * [1 / len(slices)]  # set hyperparameter to be uniform

    # Possible to pass in an initial pi (as keyword argument pis_init)
    # to improve performance, see Tutorial.ipynb notebook for more details.
    init_pis = [pst.match_spots_using_spatial_heuristic(np.array(initial_slice.obsm['spatial']),
                                                       slices[i].obsm['spatial'], use_ot=True) for i in range(4)]
    start = time.time()
    center_slice, pis = pst.center_align(initial_slice, slices, lmbda, random_seed=5, norm=True, verbose=True,
                                         pis_init=init_pis, backend=ot.backend.TorchBackend(), use_gpu=True)
    print('Integration Runtime: ' + str(time.time() - start))

    # save results
    center_slice.write('../../results/DLPFC/paste_integration_center_Sample{}_DLPFC.h5ad'.format(sample_map[sample_choose]))
    pis_dict = {}
    for i, matrix in enumerate(pis):
        pis_dict[f'pi_{i}'] = matrix
    np.savez('../../results/DLPFC/paste_integration_pis_Sample{}_DLPFC.npz'.format(sample_map[sample_choose]), **pis_dict)

    # loaded_data = np.load('matrices.npz')

    plt.rcParams.update({'font.size': 18})

    if sample_choose == 2:
        init_pis = [pst.match_spots_using_spatial_heuristic(np.array(initial_slice.obsm['spatial']),
                                                            slices[i].obsm['spatial'], use_ot=True) for i in range(4)]
        center_layer = sc.read_h5ad('../../results/DLPFC/paste_integration_center_Sample{}_DLPFC.h5ad'.format(sample_map[sample_choose]))
        B = layer_groups[sample_choose][1].shape[0] * sum(
            [0.25 * np.dot(init_pis[i], to_dense_array(layer_groups[sample_choose][i][:, center_layer.var_names].X)) for i in
             range(4)])
        center_layer_ = layer_groups[sample_choose][1].copy()[:, center_layer.var_names]
        center_layer_.X = B

        for gene in ['MFGE8', 'MOBP','PCP4','TRABD2A']:
            vmin = None
            vmax = None
            layer_idx = None
            print('initial layer')
            plot_2d_expression(layer_groups[sample_choose][1], gene, adatas, sample_list, name="initial_layer_{0}".format(gene),
                               vmin=vmin, vmax=vmax, title='{} expression in a single slice'.format(gene),
                               layer_idx=layer_idx)
            print('center layer')
            plot_2d_expression(center_layer, gene, adatas, sample_list, name="center_layer_{0}".format(gene), vmin=vmin,
                               vmax=vmax, title='{} expression in PASTE integrated slice'.format(gene), layer_idx=layer_idx)


        init_pis = [pst.match_spots_using_spatial_heuristic(np.array(initial_slice.obsm['spatial']),
                                                            slices[i].obsm['spatial'], use_ot=True) for i in range(4)]
        center_layer = sc.read_h5ad(
            '../../results/DLPFC/paste_integration_center_Sample{}_DLPFC.h5ad'.format(sample_map[sample_choose]))
        B = layer_groups[sample_choose][1].shape[0] * sum(
            [0.25 * np.dot(init_pis[i], to_dense_array(layer_groups[sample_choose][i][:, center_layer.var_names].X)) for
             i in
             range(4)])
        center_layer_ = layer_groups[sample_choose][1].copy()[:, center_layer.var_names]
        center_layer_.X = B
        plt.rcParams.update({'font.size': 22})
        # Cluster and plot raw slices using only PCA with no NMF
        for s in sample_groups[sample_choose]:
            adata = adatas[s]
            print(s)
            cluster_adata(adata, 7, sample_name=s)
            draw_spatial(adata, 'my_clusters',
                         {sample_groups[sample_choose][0]: 'Slice A', sample_groups[sample_choose][1]: 'Slice B',
                          sample_groups[sample_choose][2]: 'Slice C', sample_groups[sample_choose][3]: 'Slice D'}[s],
                         draw_contours=True)

        W = center_layer.uns['paste_W']
        H = center_layer.uns['paste_H']

        W = (W.T / W.sum(axis=1)).T
        paste_cluster_labels = KMeans(n_clusters=7, random_state=0, n_init=500).fit_predict(W)
        center_adata = adatas[sample_groups[sample_choose][1]].copy()
        center_adata = center_adata[:, center_layer.var_names]

        center_adata.X = center_layer.X
        center_layer.obs['clustering_W'] = pd.Series(
            1 + match_cluster_labels(center_adata.obs['layer_guess_reordered'], paste_cluster_labels),
            index=center_adata.obs.index, dtype='category')
        draw_spatial(center_layer, "clustering_W", 'Center H', draw_contours=True)

    # The low dimensional representation of our center slice is held
    # in the matrices W and H, which can be used for downstream analyses
    # load results
    center_slice = sc.read_h5ad('../../results/DLPFC/paste_integration_center_Sample{}_DLPFC.h5ad'.format(sample_map[sample_choose]))
    W = center_slice.uns['paste_W']
    H = center_slice.uns['paste_H']
    # print(W.shape, H.shape)
    pis_dict = np.load('../../results/DLPFC/paste_integration_pis_Sample{}_DLPFC.npz'.format(sample_map[sample_choose]))

    all_slices = pst.stack_slices_center(center_slice, slices, [pis_dict['pi_0'], pis_dict['pi_1'], pis_dict['pi_2'], pis_dict['pi_3']])

    # plot paste_integration results
    plt.figure(figsize=(7, 7))
    for i in range(len(layer_groups[sample_choose])):
        adata = all_slices[1][i]
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
    save_path = "../../results/DLPFC/paste_integration_Sample{}_DLPFC.png".format(sample_map[sample_choose])
    plt.savefig(save_path)
plt.show()


