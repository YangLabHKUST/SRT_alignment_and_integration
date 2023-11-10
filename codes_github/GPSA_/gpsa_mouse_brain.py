import anndata
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import ot
import torch
from gpsa import VariationalGPSA
from gpsa import matern12_kernel, rbf_kernel
from gpsa.plotting import callback_twod, callback_twod_aligned_only
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import r2_score
import paste as pst
import pandas as pd


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata


N_GENES = 10
N_SAMPLES = None

n_spatial_dims = 2
n_views = 2
m_G = 200
m_X_per_view = 200

N_LATENT_GPS = {"expression": None}

N_EPOCHS = 5000
PRINT_EVERY = 50

fn1 = '../../data/mouse_brain/sample1'
fn2 = '../../data/mouse_brain/sample2'
data_slice1 = sc.read_visium(fn1)
data_slice1 = process_data(data_slice1, n_top_genes=6000)
data_slice2 = sc.read_visium(fn2)
data_slice2 = process_data(data_slice2, n_top_genes=6000)

data = data_slice1.concatenate(data_slice2)

shared_gene_names = data.var.gene_ids.index.values
data_knn = data_slice1[:, shared_gene_names]
X_knn = data_knn.obsm["spatial"]
Y_knn = np.array(data_knn.X.todense())  # [:, :1000]
nbrs = NearestNeighbors(n_neighbors=2).fit(X_knn)
distances, indices = nbrs.kneighbors(X_knn)

preds = Y_knn[indices[:, 1]]
r2_vals = r2_score(Y_knn, preds, multioutput="raw_values")

gene_idx_to_keep = np.where(r2_vals > 0.1)[0]
N_GENES = min(N_GENES, len(gene_idx_to_keep))
gene_names_to_keep = data_knn.var.gene_ids.index.values[gene_idx_to_keep]
gene_names_to_keep = gene_names_to_keep[np.argsort(-r2_vals[gene_idx_to_keep])]
if N_GENES < len(gene_names_to_keep):
    gene_names_to_keep = gene_names_to_keep[:N_GENES]
data = data[:, gene_names_to_keep]
print(data)

n_samples_list = [data_slice1.shape[0], data_slice2.shape[0]]
view_idx = [
    np.arange(data_slice1.shape[0]),
    np.arange(data_slice1.shape[0], data_slice1.shape[0] + data_slice2.shape[0]),
]

pi0 = pst.match_spots_using_spatial_heuristic(data[view_idx[0]].obsm['spatial'], data[view_idx[1]].obsm['spatial'],
                                              use_ot=True)
pi12 = pst.pairwise_align(data[view_idx[0]], data[view_idx[1]], G_init=pi0, backend=ot.backend.TorchBackend(),
                          use_gpu=True)
slices, pis = [data[view_idx[0]], data[view_idx[1]]], [pi12]
new_slices = pst.stack_slices_pairwise(slices, pis)
data = anndata.concat(new_slices)

X1 = data[data.obs.batch == "0"].obsm["spatial"]
X2 = data[data.obs.batch == "1"].obsm["spatial"]
Y1 = np.array(data[data.obs.batch == "0"].X.todense())
Y2 = np.array(data[data.obs.batch == "1"].X.todense())

X1 = scale_spatial_coords(X1)
X2 = scale_spatial_coords(X2)

Y1 = (Y1 - Y1.mean(0)) / Y1.std(0)
Y2 = (Y2 - Y2.mean(0)) / Y2.std(0)

X = np.concatenate([X1, X2])
Y = np.concatenate([Y1, Y2])
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    fixed_view_idx=0,
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
fig = plt.figure(figsize=(15, 5), facecolor="white", constrained_layout=True)
data_expression_ax = fig.add_subplot(131, frameon=False)
latent_expression_ax = fig.add_subplot(132, frameon=False)
diff_expression_ax = fig.add_subplot(133, frameon=False)
plt.show(block=False)


# gene_idx = np.where(data.var.gene_ids.index.values == "Ptgds")[0]
gene_idx = 0

pd.DataFrame(view_idx["expression"]).to_csv("../../results/mouse_brain/gpsa_view_idx_mouse_brain.csv")
pd.DataFrame(X).to_csv("../../results/mouse_brain/gpsa_X_mouse_brain.csv")
pd.DataFrame(Y).to_csv("../../results/mouse_brain/gpsa_Y_mouse_brain.csv")
data.write("../../results/mouse_brain/gpsa_data_mouse_brain.h5")


for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)

    if t % PRINT_EVERY == 0 or t == N_EPOCHS - 1:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)
        diff_expression_ax.cla()

        callback_twod_aligned_only(
            model,
            X,
            Y,
            latent_expression_ax1=data_expression_ax,
            latent_expression_ax2=latent_expression_ax,
            X_aligned=G_means,
            gene_idx=gene_idx,
        )

        curr_aligned_coords = G_means["expression"].detach().cpu().numpy()

        X_knn = curr_aligned_coords[view_idx["expression"][0]]
        Y_knn = Y[view_idx["expression"][0]]
        nbrs = NearestNeighbors(n_neighbors=2).fit(X_knn)
        distances, indices = nbrs.kneighbors(
            curr_aligned_coords[view_idx["expression"][1]]
        )

        Y2_smoothed = Y_knn[indices[:, 1]]
        # import ipdb; ipdb.set_trace()
        r2_val = r2_score(Y[view_idx["expression"][1]], Y2_smoothed)
        print(r2_val, flush=True)

        Y_diffs = Y[view_idx["expression"][1]] - Y2_smoothed

        # print(np.nanmean(Y_diffs ** 2), flush=True)

        data_expression_ax.set_axis_off()
        latent_expression_ax.set_axis_off()
        diff_expression_ax.scatter(
            curr_aligned_coords[view_idx["expression"][1]][:, 0],
            curr_aligned_coords[view_idx["expression"][1]][:, 1],
            c=Y_diffs[:, gene_idx].ravel(),
            cmap="bwr",
            s=24,
            marker="H",
        )

        plt.draw()
        # plt.savefig("../../results/mouse_brain/gpsa_mouse_brain_obs_align_diff.png")
        plt.pause(1 / 60.0)

        # pd.DataFrame(curr_aligned_coords).to_csv("../../results/mouse_brain/gpsa_aligned_coords_mouse_brain.csv")

        # import ipdb; ipdb.set_trace()


plt.close()
import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

fig = plt.figure(figsize=(14, 7))
data_expression_ax = fig.add_subplot(121, frameon=False)
latent_expression_ax = fig.add_subplot(122, frameon=False)
callback_twod(
    model,
    X,
    Y,
    s=40,
    data_expression_ax=data_expression_ax,
    latent_expression_ax=latent_expression_ax,
    X_aligned=G_means,
)
latent_expression_ax.set_title("Aligned data, GPSA")
latent_expression_ax.set_axis_off()
data_expression_ax.set_axis_off()
# plt.axis("off")

plt.tight_layout()
# plt.savefig("../../results/mouse_brain/gpsa_mouse_brain.png")
plt.show()
plt.close()
