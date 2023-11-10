import seaborn as sns
import STAligner
from src.paste2.model_selection import *
from src.paste2.helper import *
import paste as pst
from utils import ICP_align
from src.paste2.projection import partial_stack_slices_pairwise

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)
slice_map = {0:'A',1:'B',2:'C',3:'D'}
sample_map = {0:'A',1:'B',2:'C'}

sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
adatas = {sample:sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
sample_groups = [["151507", "151508", "151509","151510"], [ "151669", "151670","151671", "151672"], [ "151673","151674", "151675", "151676"]]
layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]
layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]


# paste alignment
point_size = 0.8
initial_elevation = 30
initial_azimuth = 150
z_scale = 5
for i in range(len(layer_groups)):
    pis_dict = np.load('../results/DLPFC/paste_alignment_pis_Sample{}_DLPFC.npz'.format(sample_map[i]))

    slices, pis = layer_groups[i], [pis_dict['pi_0'], pis_dict['pi_1'], pis_dict['pi_2']]
    new_slices = pst.stack_slices_pairwise(slices, pis)
    print('finished stacking')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lists = [[], [], [], [], []]
    dataframes = []
    for j, L in enumerate(new_slices):
        for x, y in L.obsm['spatial']:
            lists[0].append(x)
            lists[1].append(y)
            lists[2].append(j * z_scale)
            lists[3].append(j)
        list = L.obs['layer_guess_reordered'].tolist()
        for label in list:
            lists[4].append(label)
        print(j)
    ax.scatter(lists[0], lists[1], lists[2], c=[layer_to_color_map[i] for i in lists[4]], s=point_size)
    print('finished scattering')
    ax.view_init(elev=initial_elevation, azim=initial_azimuth)
    ax.set_title('PASTE Sample ' + sample_map[i])
    save_path = f"../results/downstream/DLPFC/paste_alignment_Sample{sample_map[i]}_3D.png"
    plt.savefig(save_path)
    plt.show()


# staligner
point_size = 0.8
initial_elevation = 30
initial_azimuth = 150
z_scale = 5
for i in range(len(layer_groups)):
    print(sample_groups[i])
    Batch_list = []
    adj_list = []
    for section_id in sample_groups[i]:
        print(section_id)
        adata = adatas[section_id]
        adata.obs.rename(columns={'layer_guess_reordered': 'Ground Truth'}, inplace=True)
        # adata.var_names_make_unique(join="++")

        # make spot name unique
        adata.obs_names = [x + '_' + section_id for x in adata.obs_names]

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

    adata = sc.read_h5ad('../results/DLPFC/staligner_Sample{}_DLPFC.h5ad'.format(sample_map[i]))
    iter_comb = [(1, 0), (2, 1), (3, 2)]
    for comb in iter_comb:
        print(comb)
        k, j = comb[0], comb[1]
        adata_target = Batch_list[k]
        adata_ref = Batch_list[j]
        slice_target = sample_groups[i][k]
        slice_ref = sample_groups[i][j]

        aligned_coor = ICP_align(adata, adata_target, adata_ref, slice_target, slice_ref)
        adata_target.obsm["spatial"] = aligned_coor

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lists = [[], [], [], [], []]
    dataframes = []
    for j, L in enumerate(Batch_list):
        for x, y in L.obsm['spatial']:
            lists[0].append(x)
            lists[1].append(y)
            lists[2].append(j * z_scale)
            lists[3].append(j)
        list = L.obs['Ground Truth'].tolist()
        for label in list:
            lists[4].append(label)
        print(j)
    ax.scatter(lists[0], lists[1], lists[2], c=[layer_to_color_map[i] for i in lists[4]], s=point_size)
    print('finished scattering')
    ax.view_init(elev=initial_elevation, azim=initial_azimuth)
    ax.set_title('STAligner Sample ' + sample_map[i])
    save_path = f"../results/downstream/DLPFC/staligner_Sample{sample_map[i]}_3D.png"
    plt.savefig(save_path)
    plt.show()


# paste2
point_size = 0.8
initial_elevation = 30
initial_azimuth = 150
z_scale = 5
for i in range(len(layer_groups)):
    pis_dict = np.load('../results/DLPFC/paste2_pis_Sample{}_DLPFC.npz'.format(sample_map[i]))

    slices, pis = layer_groups[i], [pis_dict['pi_0'], pis_dict['pi_1'], pis_dict['pi_2']]
    new_slices = partial_stack_slices_pairwise(slices, pis)
    print('finished stacking')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lists = [[], [], [], [], []]
    dataframes = []
    for j, L in enumerate(new_slices):
        for x, y in L.obsm['spatial']:
            lists[0].append(x)
            lists[1].append(y)
            lists[2].append(j * z_scale)
            lists[3].append(j)
        list = L.obs['layer_guess_reordered'].tolist()
        for label in list:
            lists[4].append(label)
        print(j)
    ax.scatter(lists[0], lists[1], lists[2], c=[layer_to_color_map[i] for i in lists[4]], s=point_size)
    print('finished scattering')
    ax.view_init(elev=initial_elevation, azim=initial_azimuth)
    ax.set_title('PASTE2 Sample ' + sample_map[i])
    save_path = f"../results/downstream/DLPFC/paste2_Sample{sample_map[i]}_3D.png"
    plt.savefig(save_path)
    plt.show()


# GPSA
point_size = 0.8
initial_elevation = 30
initial_azimuth = 150
z_scale = 5
spots_count = [[0], [0], [0]]
for i in range(len(sample_groups)):
    n = 0
    for sample in sample_groups[i]:
        num = adatas[sample].shape[0]
        n += num
        spots_count[i].append(n)

for i in range(len(layer_groups)):
    adata = sc.read_csv("../results/DLPFC/gpsa_Sample{}_DLPFC_aligned_coords_st.csv".format(sample_map[i])).X[1:]
    view_idx = [np.arange(spots_count[i][ii], spots_count[i][ii + 1]) for ii in range(len(layer_groups[i]))]
    coords_slice1 = adata[view_idx[0]]
    coords_slice2 = adata[view_idx[1]]
    coords_slice3 = adata[view_idx[2]]
    coords_slice4 = adata[view_idx[3]]
    coords = [coords_slice1, coords_slice2, coords_slice3, coords_slice4]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lists = [[], [], [], [], []]
    dataframes = []
    for j, L in enumerate(coords):
        for x, y in L:
            lists[0].append(x)
            lists[1].append(y)
            lists[2].append(j * z_scale)
            lists[3].append(j)
        list = layer_groups[i][j].obs['layer_guess_reordered'].tolist()
        for label in list:
            lists[4].append(label)
        print(j)
    ax.scatter(lists[0], lists[1], lists[2], c=[layer_to_color_map[i] for i in lists[4]], s=point_size)
    print('finished scattering')
    ax.view_init(elev=initial_elevation, azim=initial_azimuth)
    ax.set_title('GPSA Sample ' + sample_map[i])
    save_path = f"../results/downstream/DLPFC/gpsa_Sample{sample_map[i]}_3D.png"
    plt.savefig(save_path)
    plt.show()