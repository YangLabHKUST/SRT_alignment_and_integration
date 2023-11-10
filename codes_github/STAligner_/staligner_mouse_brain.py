import matplotlib
from ..utils import *
import matplotlib.pyplot as plt
import torch
import STAligner
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fn1 = '../../data/mouse_brain/sample1'
fn2 = '../../data/mouse_brain/sample2'
slice1 = sc.read_visium(fn1)
slice1.var_names_make_unique()
slice1 = process_data(slice1, n_top_genes=5000)
slice2 = sc.read_visium(fn2)
slice1.var_names_make_unique()
slice2 = process_data(slice2, n_top_genes=5000)
slices = {'slice1': slice1, 'slice2': slice2}

sample_list = ['slice1', 'slice2']
Batch_list = []
adj_list = []
for section_id in sample_list:
    print(section_id)
    adata = slices[section_id]
    # adata.var_names_make_unique(join="++")

    # make spot name unique
    adata.obs_names = [x+'_'+section_id for x in adata.obs_names]

    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=200)  # the spatial network are saved in adata.uns[‘adj’]
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

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(sample_list)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=50, device=device)
edge_list = [[left, right] for left, right in zip(adata_concat.uns['edgeList'][0], adata_concat.uns['edgeList'][1])]
adata_concat.uns['edgeList'] = edge_list
adata_concat.write('../../results/mouse_brain/staligner_mouse_brain.h5ad')

adata = sc.read_h5ad('../../results/mouse_brain/staligner_mouse_brain.h5ad')
comb = (1,0)
print(comb)
k, j = comb[0], comb[1]
adata_target = Batch_list[k]
adata_ref = Batch_list[j]
slice_target = sample_list[k]
slice_ref = sample_list[j]

aligned_coor = ICP_align(adata, adata_target, adata_ref, slice_target, slice_ref)
adata_target.obsm["spatial"] = aligned_coor

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

fig = plt.figure(figsize=(14, 7))
data_expression_ax = fig.add_subplot(121, frameon=False)
latent_expression_ax = fig.add_subplot(122, frameon=False)
idx = (slice1.concatenate(slice2)).var_names.tolist().index('Pcp2')
callback([slice1, slice2], Batch_list, s=40, data_expression_ax=data_expression_ax,
         latent_expression_ax=latent_expression_ax, gene_idx=idx)
latent_expression_ax.set_title("Aligned data, STAligner")
latent_expression_ax.set_axis_off()
data_expression_ax.set_axis_off()

plt.tight_layout()
plt.savefig("../../results/mouse_brain/staligner_mouse_brain.png")
plt.show()