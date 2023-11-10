import matplotlib.patches as mpatches
import paste as pst
import torch
import STAligner
from ..utils import *
import warnings
warnings.filterwarnings("ignore")


# Load Slices
data_dir = '../../data/Stahl-BC/'  # change this path to the data you wish to analyze


# Assume that the coordinates of slices are named slice_name + "_coor.csv"
def load_slices(data_dir, slice_names=["slice1", "slice2"]):
    slices = []
    for slice_name in slice_names:
        slice_i = sc.read_csv(data_dir + slice_name + ".csv")
        slice_i_coor = np.genfromtxt(data_dir + slice_name + "_coor.csv", delimiter = ',')
        slice_i.obsm['spatial'] = slice_i_coor
        # Preprocess slices
        sc.pp.filter_genes(slice_i, min_counts=15)
        sc.pp.filter_cells(slice_i, min_counts=100)
        slice_i.X = scipy.sparse.csr_matrix(slice_i.X)
        slices.append(slice_i)
    return slices


slice_names = ['stahl_bc_slice1', 'stahl_bc_slice2', 'stahl_bc_slice3', 'stahl_bc_slice4']
slices = load_slices(data_dir, slice_names=slice_names)
slice1, slice2, slice3, slice4 = slices
slices = {'slice1': slice1, 'slice2': slice2, 'slice3': slice3, 'slice4': slice4}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample_list = ['slice1', 'slice2', 'slice3', 'slice4']
Batch_list = []
adj_list = []
for section_id in sample_list:
    print(section_id)
    adata = slices[section_id]
    adata.var_names_make_unique(join="++")

    # make spot name unique
    adata.obs_names = [x+'_'+section_id for x in adata.obs_names]

    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=2)  # the spatial network are saved in adata.uns[‘adj’]
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
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)
print(adata_concat)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(sample_list)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=50, device=device)
print(adata_concat.obsm['STAligner'].shape)
edge_list = [[left, right] for left, right in zip(adata_concat.uns['edgeList'][0], adata_concat.uns['edgeList'][1])]
adata_concat.uns['edgeList'] = edge_list
adata_concat.write('../../results/stahl_bc/staligner_stahl_bc.h5ad')

adata = sc.read_h5ad('../../results/stahl_bc/staligner_stahl_bc.h5ad')
iter_comb = [(1,0), (2,1), (3,2)]
for comb in iter_comb:
    print(comb)
    i, j = comb[0], comb[1]
    adata_target = Batch_list[i]
    adata_ref = Batch_list[j]
    slice_target = sample_list[i]
    slice_ref = sample_list[j]

    aligned_coor = ICP_align(adata, adata_target, adata_ref, slice_target, slice_ref)
    adata_target.obsm["spatial"] = aligned_coor

slice_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']
plt.figure(figsize=(7,7))
for i in range(len(Batch_list)):
    pst.plot_slice(Batch_list[i],slice_colors[i], s=400)
plt.legend(handles=[mpatches.Patch(color=slice_colors[0], label='1'),
                    mpatches.Patch(color=slice_colors[1], label='2'),
                    mpatches.Patch(color=slice_colors[2], label='3'),
                    mpatches.Patch(color=slice_colors[3], label='4')])
plt.gca().invert_yaxis()
plt.axis('off')
# save_path = "../../results/stahl_bc/staligner_stahl_bc.png"
# plt.savefig(save_path)
plt.show()




