import STAligner
import scipy.linalg
import matplotlib.patches as mpatches
import torch
from codes_github.utils import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
adatas = {sample:sc.read_h5ad('../../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
sample_groups = [["151507", "151508", "151509","151510"], [ "151669", "151670","151671", "151672"], [ "151673","151674", "151675", "151676"]]
layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]
layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]

slice_map = {0:'A',1:'B',2:'C',3:'D'}
sample_map = {0:'A',1:'B',2:'C'}


for i in range(3):
    print(sample_groups[i])
    Batch_list = []
    adj_list = []
    for section_id in sample_groups[i]:
        print(section_id)
        adata = adatas[section_id]
        adata.obs.rename(columns={'layer_guess_reordered': 'Ground Truth'}, inplace=True)
        # adata.var_names_make_unique(join="++")

        # make spot name unique
        adata.obs_names = [x+'_'+section_id for x in adata.obs_names]

        # Constructing the spatial network
        STAligner.Cal_Spatial_Net(adata, rad_cutoff=10)  # the spatial network are saved in adata.uns[‘adj’]
        # STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors

        # Normalization
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, adata.var['highly_variable']]

        adj_list.append(adata.uns['adj'])
        Batch_list.append(adata)

    # concat the scanpy objects of multiple slices
    adata_concat = ad.concat(Batch_list, label="slice_name", keys=sample_groups[i])
    adata_concat.obs['Ground Truth'] = adata_concat.obs['Ground Truth'].astype('category')
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
    print('adata_concat.shape: ', adata_concat.shape)

    # concat the spatial networks of multiple slices
    adj_concat = np.asarray(adj_list[0].todense())
    for batch_id in range(1, len(sample_groups[i])):
        adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
    adata_concat.uns['edgeList'] = np.nonzero(adj_concat)
    # print(adata_concat.uns['edgeList'])

    adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=50, device=device)
    # print(adata_concat)
    # del adata_concat.uns['edgeList']
    edge_list = [[left, right] for left, right in zip(adata_concat.uns['edgeList'][0], adata_concat.uns['edgeList'][1])]
    adata_concat.uns['edgeList'] = edge_list
    adata_concat.write('../../results/DLPFC/staligner_Sample{}_DLPFC.h5ad'.format(sample_map[i]))

    # # clustering
    # if i == 1:
    #     STAligner.ST_utils.mclust_R(adata_concat, num_cluster=5, used_obsm='STAligner')
    # else:
    #     STAligner.ST_utils.mclust_R(adata_concat, num_cluster=7, used_obsm='STAligner')
    # adata_concat = adata_concat[adata_concat.obs['Ground Truth']!='unknown']
    # print('mclust, ARI = %01.3f' % ari_score(adata_concat.obs['Ground Truth'], adata_concat.obs['mclust']))

    adata = sc.read_h5ad('../../results/DLPFC/staligner_Sample{}_DLPFC.h5ad'.format(sample_map[i]))
    iter_comb = [(1,0), (2,1), (3,2)]
    for comb in iter_comb:
        print(comb)
        k, j = comb[0], comb[1]
        adata_target = Batch_list[k]
        adata_ref = Batch_list[j]
        slice_target = sample_groups[i][k]
        slice_ref = sample_groups[i][j]

        aligned_coor = ICP_align(adata, adata_target, adata_ref, slice_target, slice_ref)
        adata_target.obsm["spatial"] = aligned_coor

    plt.figure(figsize=(7, 7))
    for m in range(len(sample_groups[i])):
        adata = Batch_list[m]
        colors = list(adata.obs['Ground Truth'].astype('str').map(layer_to_color_map))
        plt.scatter(adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1], linewidth=0, s=70, marker=".",
                    color=colors)
    plt.title('Sample' + sample_map[i], size=12)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.legend(handles=[mpatches.Patch(color=layer_to_color_map[adata.obs['Ground Truth'].cat.categories[i]],
                                       label=adata.obs['Ground Truth'].cat.categories[i]) for i in
                        range(len(adata.obs['Ground Truth'].cat.categories))], fontsize=6,
               title='Cortex layer', title_fontsize=6, bbox_to_anchor=(1, 1))
    save_path = "../../results/DLPFC/staligner_Sample{}_DLPFC.png".format(sample_map[i])
    plt.savefig(save_path)
plt.show()


# joint slices
sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
adatas = {sample:sc.read_h5ad('../../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]

slice_map = {0:'A',1:'B',2:'C',3:'D'}
sample_map = {0:'A',1:'B',2:'C'}
Batch_list = []
adj_list = []
for section_id in sample_list:
    print(section_id)
    adata = adatas[section_id]
    adata.obs.rename(columns={'layer_guess_reordered': 'Ground Truth'}, inplace=True)
    # adata.var_names_make_unique(join="++")

    # make spot name unique
    adata.obs_names = [x+'_'+section_id for x in adata.obs_names]

    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=10)  # the spatial network are saved in adata.uns[‘adj’]
    # STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors

    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]

    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)

# concat the scanpy objects of multiple slices
adata_concat = ad.concat(Batch_list, label="slice_name", keys=sample_list)
adata_concat.obs['Ground Truth'] = adata_concat.obs['Ground Truth'].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)

# combine slice names into sample name
new_batch_1 = adata_concat.obs["slice_name"].isin(['151507', '151508', '151509', '151510'])
new_batch_2 = adata_concat.obs["slice_name"].isin(['151669', '151670', '151671', '151672'])
new_batch_3 = adata_concat.obs["slice_name"].isin(['151673', '151674', '151675', '151676'])
adata_concat.obs["sample_name"] = list(sum(new_batch_1)*['Sample A'])+list(sum(new_batch_2)*['Sample B'])+list(sum(new_batch_3)*['Sample C'])
adata_concat.obs["sample_name"] = adata_concat.obs["sample_name"].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["sample_name"].copy()

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(sample_list)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

adata_concat = STAligner.train_STAligner(adata_concat, iter_comb=None, verbose=True, device=device, margin=1.0)
print(adata_concat.obsm['STAligner'].shape)
edge_list = [[left, right] for left, right in zip(adata_concat.uns['edgeList'][0], adata_concat.uns['edgeList'][1])]
adata_concat.uns['edgeList'] = edge_list
adata_concat.write('../../results/DLPFC/staligner_Sample_all_DLPFC.h5ad')



