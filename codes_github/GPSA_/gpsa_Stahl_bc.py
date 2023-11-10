import anndata
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import ot
import time
import torch
from gpsa import VariationalGPSA
from gpsa import matern12_kernel, rbf_kernel
from gpsa.plotting import callback_twod
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import r2_score
import paste as pst

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Slices
data_dir = '../../data/Stahl-BC/'  # change this path to the data you wish to analyze
N_GENES = 20
N_SAMPLES = None
N_LAYERS = 4
fixed_view_idx = 1

n_spatial_dims = 2
n_views = 4
m_G = 200
m_X_per_view = 200

N_LATENT_GPS = {"expression": None}

N_EPOCHS = 5000
PRINT_EVERY = 50


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


# Assume that the coordinates of slices are named slice_name + "_coor.csv"
def load_slices(data_dir, slice_names=["slice1", "slice2"]):
    slices = []
    for slice_name in slice_names:
        slice_i = sc.read_csv(data_dir + slice_name + ".csv")
        slice_i_coor = np.genfromtxt(data_dir + slice_name + "_coor.csv", delimiter = ',')
        slice_i.obsm['spatial'] = slice_i_coor
        # Preprocess slices
        # sc.pp.filter_genes(slice_i, min_counts=15)
        # sc.pp.filter_cells(slice_i, min_counts=100)
        slices.append(slice_i)

    return slices


def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    sc.pp.filter_cells(adata, min_counts=100)
    # sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    # sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata


# load slices
slice_names = ['stahl_bc_slice1', 'stahl_bc_slice2', 'stahl_bc_slice3', 'stahl_bc_slice4']
slices = load_slices(data_dir, slice_names=slice_names)
slice1, slice2, slice3, slice4 = slices


# filtered image
slice_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']

process_data(slice1, n_top_genes=3000)
process_data(slice2, n_top_genes=3000)
process_data(slice3, n_top_genes=3000)
process_data(slice4, n_top_genes=3000)

data = anndata.concat([slice1,slice2,slice3,slice4])

shared_gene_names = data.var.index.values
data_knn = slice1[:, shared_gene_names]
X_knn = data_knn.obsm["spatial"]
Y_knn = data_knn.X
Y_knn = (Y_knn - Y_knn.mean(0)) / Y_knn.std(0)
# nbrs = NearestNeighbors(n_neighbors=2).fit(X_knn)
# distances, indices = nbrs.kneighbors(X_knn)
knn = KNeighborsRegressor(n_neighbors=10, weights="uniform").fit(X_knn, Y_knn)
preds = knn.predict(X_knn)

r2_vals = r2_score(Y_knn, preds, multioutput="raw_values")

gene_idx_to_keep = np.where(r2_vals > 0.3)[0]
print(gene_idx_to_keep)
N_GENES = min(N_GENES, len(gene_idx_to_keep))
gene_names_to_keep = data_knn.var.index.values[gene_idx_to_keep]
gene_names_to_keep = gene_names_to_keep[np.argsort(-r2_vals[gene_idx_to_keep])]
r2_vals_sorted = -1 * np.sort(-r2_vals[gene_idx_to_keep])
if N_GENES < len(gene_names_to_keep):
    gene_names_to_keep = gene_names_to_keep[:N_GENES]
data = data[:, gene_names_to_keep]

n_samples_list = [
    slice1.shape[0],
    slice2.shape[0],
    slice3.shape[0],
    slice4.shape[0],
]
cumulative_sum = np.cumsum(n_samples_list)
cumulative_sum = np.insert(cumulative_sum, 0, 0)
view_idx = [
    np.arange(cumulative_sum[ii], cumulative_sum[ii + 1]) for ii in range(n_views)
]

# Pairwise align the slices
start = time.time()
pi12 = pst.pairwise_align(data[view_idx[0]], data[view_idx[1]])#, backend=ot.backend.TorchBackend(), use_gpu=True)
pi23 = pst.pairwise_align(data[view_idx[1]], data[view_idx[2]])#, backend=ot.backend.TorchBackend(), use_gpu=True)
pi34 = pst.pairwise_align(data[view_idx[2]], data[view_idx[3]])#, backend=ot.backend.TorchBackend(), use_gpu=True)
print('Alignment Runtime: ' + str(time.time() - start))
slices, pis = [data[view_idx[0]], data[view_idx[1]], data[view_idx[2]], data[view_idx[3]]], [pi12, pi23, pi34]
new_slices = pst.stack_slices_pairwise(slices, pis)
data = anndata.concat(new_slices)

X_list = []
Y_list = []
for vv in range(n_views):
    curr_X = np.array(data[view_idx[vv]].obsm["spatial"])
    curr_Y = data[view_idx[vv]].X

    curr_X = scale_spatial_coords(curr_X)
    curr_Y = (curr_Y - curr_Y.mean(0)) / curr_Y.std(0)

    X_list.append(curr_X)
    Y_list.append(curr_Y)


X = np.concatenate(X_list)
Y = np.concatenate(Y_list)

n_outputs = Y.shape[1]

x = torch.from_numpy(X).float().clone().to(device)
y = torch.from_numpy(Y).float().clone().to(device)

data_dict = {
    "expression": {
        "spatial_coords": x,
        "outputs": y,
        "n_samples_list": n_samples_list,
    }
}

model = VariationalGPSA(
    data_dict,
    n_spatial_dims=n_spatial_dims,
    m_X_per_view=m_X_per_view,
    m_G=m_G,
    data_init=True,
    minmax_init=False,
    grid_init=False,
    n_latent_gps=N_LATENT_GPS,
    mean_function="identity_fixed",
    kernel_func_warp=rbf_kernel,
    kernel_func_data=rbf_kernel,
    # fixed_warp_kernel_variances=np.ones(n_views) * 1.,
    # fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
    fixed_view_idx=fixed_view_idx,
).to(device)

view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


def train(model, loss_fn, optimizer):
    model.train()

    # Forward pass
    G_means, G_samples, F_latent_samples, F_samples = model.forward(
        X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns, S=5
    )

    # Compute loss
    loss = loss_fn(data_dict, F_samples)

    # Compute gradients and take optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), G_means


# Set up figure.
fig = plt.figure(figsize=(10, 5), facecolor="white", constrained_layout=True)
ax1 = fig.add_subplot(121, frameon=False)
ax2 = fig.add_subplot(122, frameon=False)
plt.show(block=False)

gene_idx = 0

for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)

    if t % PRINT_EVERY == 0 or t == N_EPOCHS - 1:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)
        ax1.cla()
        ax2.cla()

        curr_aligned_coords = G_means["expression"].detach().cpu().numpy()
        curr_aligned_coords_slice1 = curr_aligned_coords[view_idx["expression"][0]]
        curr_aligned_coords_slice2 = curr_aligned_coords[view_idx["expression"][1]]
        curr_aligned_coords_slice3 = curr_aligned_coords[view_idx["expression"][2]]
        curr_aligned_coords_slice4 = curr_aligned_coords[view_idx["expression"][3]]

        for vv, curr_X in enumerate(X_list):
            if vv == fixed_view_idx:
                ax1.scatter(curr_X[:, 0], curr_X[:, 1],linewidth=0, s=80, alpha=1, color='black')
                ax2.scatter(
                    X_list[vv][:, 0], X_list[vv][:, 1],linewidth=0, s=80, alpha=1, color='black'
                )

            ax1.scatter(curr_X[:, 0], curr_X[:, 1], linewidth=0, s=80, alpha=1, color=slice_colors[vv])
            ax2.scatter(
                curr_aligned_coords[view_idx["expression"][vv]][:, 0],
                curr_aligned_coords[view_idx["expression"][vv]][:, 1],
                linewidth=0, s=80, alpha=1, color=slice_colors[vv]
            )

        plt.legend(handles=[mpatches.Patch(color=slice_colors[0], label='1'),
                            mpatches.Patch(color=slice_colors[1], label='2'),
                            mpatches.Patch(color=slice_colors[2], label='3'),
                            mpatches.Patch(color=slice_colors[3], label='4')])
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax1.axis('off')
        ax2.axis('off')
        plt.draw()
        plt.savefig("../../results/stahl_bc/gpsa_stahl_bc_without_PASTE.png")
        plt.pause(1 / 60.0)

        if t == N_EPOCHS - 1:
            fig, axs = plt.subplots(2, 2, figsize=(7, 7))
            for vv, curr_X in enumerate(X_list):
                axs[int(vv / 2), int(vv % 2)].scatter(
                    curr_aligned_coords[view_idx["expression"][fixed_view_idx]][:, 0],
                    curr_aligned_coords[view_idx["expression"][fixed_view_idx]][:, 1],
                    linewidth=0, s=20, alpha=1, color='orange'
                )
                axs[int(vv / 2), int(vv % 2)].scatter(
                    curr_aligned_coords[view_idx["expression"][vv]][:, 0],
                    curr_aligned_coords[view_idx["expression"][vv]][:, 1],
                    linewidth=0, s=20, alpha=1, color=slice_colors[vv]
                )
                axs[int(vv / 2), int(vv % 2)].axis('off')
            # save_path = "../../results/stahl_bc/gpsa_template_individual_stahl_bc.png"
            # plt.savefig(save_path)
            plt.show()

            fig, axs = plt.subplots(2, 2, figsize=(7, 7))
            for vv, curr_X in enumerate(X_list):
                if vv == N_LAYERS - 1:
                    break
                axs[int(vv / 2), int(vv % 2)].scatter(
                    curr_aligned_coords[view_idx["expression"][vv]][:, 0],
                    curr_aligned_coords[view_idx["expression"][vv]][:, 1],
                    linewidth=0, s=20, alpha=1, color=slice_colors[vv]
                )
                axs[int(vv / 2), int(vv % 2)].scatter(
                    curr_aligned_coords[view_idx["expression"][vv + 1]][:, 0],
                    curr_aligned_coords[view_idx["expression"][vv + 1]][:, 1],
                    linewidth=0, s=20, alpha=1, color=slice_colors[vv + 1]
                )
                axs[int(vv / 2), int(vv % 2)].axis('off')
            fig.delaxes(axs[1, 1])
            # save_path = "../../results/stahl_bc/gpsa_sequence_individual_stahl_bc.png"
            # plt.savefig(save_path)
            plt.show()
