import seaborn as sns
import umap
from sklearn import preprocessing
from matplotlib.colors import ListedColormap
from paste2.model_selection import *
from paste2.helper import *

import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)
slice_map = {0:'A',1:'B',2:'C',3:'D'}
sample_map = {0:'A',1:'B',2:'C'}


# umap
reducer = umap.UMAP(n_neighbors=30,
                    n_components=2,
                    metric="correlation",
                    n_epochs=None,
                    learning_rate=1.0,
                    min_dist=0.3,
                    spread=1.0,
                    set_op_mix_ratio=1.0,
                    local_connectivity=1,
                    repulsion_strength=1,
                    negative_sample_rate=5,
                    a=None,
                    b=None,
                    random_state=1234,
                    metric_kwds=None,
                    angular_rp_forest=False,
                    verbose=True)

sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
adatas = {sample:sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
sample_groups = [["151507", "151508", "151509","151510"], [ "151669", "151670","151671", "151672"], [ "151673","151674", "151675", "151676"]]
layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]
layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]


# staligner sample specific
for group in range(len(layer_groups)):
    print('sample' + sample_map[group])
    adata = sc.read_h5ad('../results/DLPFC/staligner_Sample{}_DLPFC.h5ad'.format(sample_map[group]))
    embedding = reducer.fit_transform(adata.obsm['STAligner'])
    adata.obsm["X_umap"] = embedding
    pd.DataFrame(adata.obsm["X_umap"]).to_csv(f"../results/downstream/DLPFC/staligner_Sample{sample_map[group]}_umap.csv")
    raw_embedding = reducer.fit_transform(adata.X)
    adata.obsm["X_umap_raw"] = raw_embedding
    pd.DataFrame(adata.obsm["X_umap_raw"]).to_csv(f"../results/downstream/DLPFC/raw_Sample{sample_map[group]}_umap.csv")

ratios = [[1.42, 1.6], [1.03, 1.3], [2.09, 1.25]]
for group in range(len(layer_groups)):
    ratio = ratios[group]
    adata = sc.read_h5ad('../results/DLPFC/staligner_Sample{}_DLPFC.h5ad'.format(sample_map[group]))
    adata.obs.rename(columns={'Ground Truth': 'layer_guess_reordered'}, inplace=True)

    clusters = sc.read_csv(f"../results/downstream/DLPFC/staligner_Sample{sample_map[group]}_clustering_result.csv").X
    adata.obs["my_clusters"] = clusters
    embedding = sc.read_csv(f"../results/downstream/DLPFC/staligner_Sample{sample_map[group]}_umap.csv").X[1:]
    adata.obsm["X_umap"] = embedding
    raw_embedding = sc.read_csv(f"../results/downstream/DLPFC/raw_Sample{sample_map[group]}_umap.csv").X[1:]
    adata.obsm["X_umap_raw"] = raw_embedding

    n_spots = embedding.shape[0]
    size = 10000 / n_spots

    sc.pp.neighbors(adata, use_rep='STAligner')
    sc.tl.paga(adata, groups='layer_guess_reordered')

    le_slice = preprocessing.LabelEncoder()
    label_slice = le_slice.fit_transform(adata.obs['slice_name'])

    le_layer = preprocessing.LabelEncoder()
    label_layer = le_layer.fit_transform(adata.obs['layer_guess_reordered'])

    order = np.arange(n_spots)
    np.random.shuffle(order)

    f = plt.figure(figsize=(22.5, 5))

    ax1 = f.add_subplot(1, 5, 1)
    ax1.set_aspect(ratio[0])
    scatter1 = ax1.scatter(raw_embedding[order, 0], raw_embedding[order, 1],
                           s=size, c=label_slice[order], cmap='coolwarm')
    ax1.set_title("Raw Slice", fontsize=15)
    ax1.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                    grid_alpha=0)

    l1 = ax1.legend(handles=scatter1.legend_elements()[0],
                    labels=["Slice %r" % i for i in sample_groups[group]],
                    loc="upper left", bbox_to_anchor=(0., 0.),
                    markerscale=1., title_fontsize=10, fontsize=10, frameon=False, ncol=1)
    l1._legend_box.align = "left"

    ax2 = f.add_subplot(1, 5, 2)
    ax2.set_aspect(ratio[1])
    scatter2 = ax2.scatter(embedding[order, 0], embedding[order, 1],
                           s=size, c=label_slice[order], cmap='coolwarm')
    ax2.set_title("Latent", fontsize=15)
    ax2.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                    grid_alpha=0)

    l2 = ax2.legend(handles=scatter2.legend_elements()[0],
                    labels=["Slice %r" % i for i in sample_groups[group]],
                    loc="upper left", bbox_to_anchor=(0., 0.),
                    markerscale=1., title_fontsize=10, fontsize=10, frameon=False, ncol=1)
    l2._legend_box.align = "left"

    ax3 = f.add_subplot(1, 5, 3)
    ax3.set_aspect(ratio[1])
    scatter3 = ax3.scatter(embedding[order, 0], embedding[order, 1],
                           s=size, c=adata.obs['my_clusters'][order], cmap='cividis')
    ax3.set_title("Cluster", fontsize=15)
    ax3.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                    grid_alpha=0)

    l3 = ax3.legend(handles=scatter3.legend_elements()[0],
                    labels=["Cluster %d" % i for i in range(1, 8)],
                    loc="upper left", bbox_to_anchor=(-0.1, 0.),
                    markerscale=1., title_fontsize=10, fontsize=10, frameon=False, ncol=2, columnspacing=0.5)
    l3._legend_box.align = "left"

    ax4 = f.add_subplot(1, 5, 4)
    ax4.set_aspect(ratio[1])
    if group == 1:
        scatter4 = ax4.scatter(embedding[order, 0], embedding[order, 1],
                               s=size, c=label_layer[order], cmap=ListedColormap(
                ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
    else:
        scatter4 = ax4.scatter(embedding[order, 0], embedding[order, 1],
                               s=size, c=label_layer[order], cmap=ListedColormap(
                ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]))
    ax4.set_title("Layer annotation", fontsize=15)
    ax4.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                    grid_alpha=0)

    l4 = ax4.legend(handles=scatter4.legend_elements()[0],
                    labels=sorted(set(adata.obs['layer_guess_reordered'].values)),
                    loc="upper left", bbox_to_anchor=(-0.05, 0.),
                    markerscale=1., title_fontsize=10, fontsize=10, frameon=False, ncol=2, columnspacing=0.5)
    l4._legend_box.align = "left"

    ax5 = f.add_subplot(1, 5, 5)
    ax5.set_aspect(ratio[1])
    ax5.set_title("Trajectory", fontsize=15)
    ax5.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                    grid_alpha=0)
    ax5.set_xlim(ax4.get_xlim())
    ax5.set_ylim(ax4.get_ylim())

    pos = []
    if group == 1:
        for layer in ["Layer3", "Layer4", "Layer5", "Layer6", "WM"]:
            center = np.mean(embedding[adata.obs['layer_guess_reordered'].values.astype(str) == layer, :], axis=0)
            pos.append(center)
    else:
        for layer in ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"]:
            center = np.mean(embedding[adata.obs['layer_guess_reordered'].values.astype(str) == layer, :], axis=0)
            pos.append(center)
    sc.pl.paga(adata, pos=np.array(pos), node_size_scale=2, edge_width_scale=1, fontsize=5, fontoutline=1, ax=ax5)

    f.subplots_adjust(hspace=.1, wspace=.1)
    save_path = f"../results/downstream/DLPFC/staligner_Sample{sample_map[group]}_umap.png"
    f.savefig(save_path)


# staligner all samples
staligner_all = sc.read_h5ad('../results/DLPFC/staligner_Sample_all_DLPFC.h5ad')
embedding = reducer.fit_transform(staligner_all.obsm['STAligner'])
staligner_all.obsm["X_umap"] = embedding
pd.DataFrame(staligner_all.obsm["X_umap"]).to_csv("../results/downstream/DLPFC/staligner_Sample_all_umap.csv")
raw_embedding = reducer.fit_transform(staligner_all.X)
staligner_all.obsm["X_umap_raw"] = raw_embedding
pd.DataFrame(staligner_all.obsm["X_umap_raw"]).to_csv("../results/downstream/DLPFC/raw_Sample_all_umap.csv")

ratio = [1.48, 1.1]
adata = sc.read_h5ad('../results/DLPFC/staligner_Sample_all_DLPFC.h5ad')
adata.obs.rename(columns={'Ground Truth': 'layer_guess_reordered'}, inplace=True)

clusters = sc.read_csv("../results/downstream/DLPFC/staligner_Sample_all_clustering_result.csv").X
adata.obs["my_clusters"] = clusters
embedding = sc.read_csv("../results/downstream/DLPFC/staligner_Sample_all_umap.csv").X[1:]
adata.obsm["X_umap"] = embedding
raw_embedding = sc.read_csv("../results/downstream/DLPFC/raw_Sample_all_umap.csv").X[1:]
adata.obsm["X_umap_raw"] = raw_embedding

n_spots = embedding.shape[0]
size = 10000 / n_spots

sc.pp.neighbors(adata, use_rep='STAligner')
sc.tl.paga(adata, groups='layer_guess_reordered')

le_slice = preprocessing.LabelEncoder()
label_slice = le_slice.fit_transform(adata.obs['batch_name'])

le_layer = preprocessing.LabelEncoder()
label_layer = le_layer.fit_transform(adata.obs['layer_guess_reordered'])

order = np.arange(n_spots)
np.random.shuffle(order)

f = plt.figure(figsize=(22.5, 5))

ax1 = f.add_subplot(1, 5, 1)
ax1.set_aspect(ratio[0])
scatter1 = ax1.scatter(raw_embedding[order, 0], raw_embedding[order, 1],
                       s=size, c=label_slice[order], cmap='coolwarm')
ax1.set_title("Raw Sample", fontsize=15)
ax1.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                grid_alpha=0)

l1 = ax1.legend(handles=scatter1.legend_elements()[0],
                labels=[f"Sample {sample_map[i]}" for i in range(3)],
                loc="upper left", bbox_to_anchor=(0., 0.),
                markerscale=1., title_fontsize=10, fontsize=10, frameon=False, ncol=1)
l1._legend_box.align = "left"

ax2 = f.add_subplot(1, 5, 2)
ax2.set_aspect(ratio[1])
scatter2 = ax2.scatter(embedding[order, 0], embedding[order, 1],
                       s=size, c=label_slice[order], cmap='coolwarm')
ax2.set_title("Latent", fontsize=15)
ax2.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                grid_alpha=0)

l2 = ax2.legend(handles=scatter2.legend_elements()[0],
                labels=[f"Sample {sample_map[i]}" for i in range(3)],
                loc="upper left", bbox_to_anchor=(0., 0.),
                markerscale=1., title_fontsize=10, fontsize=10, frameon=False, ncol=1)
l2._legend_box.align = "left"

ax3 = f.add_subplot(1, 5, 3)
ax3.set_aspect(ratio[1])
scatter3 = ax3.scatter(embedding[order, 0], embedding[order, 1],
                       s=size, c=adata.obs['my_clusters'][order], cmap='cividis')
ax3.set_title("Cluster", fontsize=15)
ax3.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                grid_alpha=0)

l3 = ax3.legend(handles=scatter3.legend_elements()[0],
                labels=["Cluster %d" % i for i in range(1, 8)],
                loc="upper left", bbox_to_anchor=(-0.1, 0.),
                markerscale=1., title_fontsize=10, fontsize=10, frameon=False, ncol=2, columnspacing=0.5)

l3._legend_box.align = "left"

ax4 = f.add_subplot(1, 5, 4)
ax4.set_aspect(ratio[1])
scatter4 = ax4.scatter(embedding[order, 0], embedding[order, 1],
                       s=size, c=label_layer[order], cmap=ListedColormap(
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]))
ax4.set_title("Layer annotation", fontsize=15)
ax4.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                grid_alpha=0)

l4 = ax4.legend(handles=scatter4.legend_elements()[0],
                labels=sorted(set(adata.obs['layer_guess_reordered'].values)),
                loc="upper left", bbox_to_anchor=(-0.05, 0.),
                markerscale=1., title_fontsize=10, fontsize=10, frameon=False, ncol=2, columnspacing=0.5)

l4._legend_box.align = "left"

ax5 = f.add_subplot(1, 5, 5)
ax5.set_aspect(ratio[1])
ax5.set_title("Trajectory", fontsize=15)
ax5.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                grid_alpha=0)
ax5.set_xlim(ax4.get_xlim())
ax5.set_ylim(ax4.get_ylim())

pos = []
for layer in ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"]:
    center = np.mean(embedding[adata.obs['layer_guess_reordered'].values.astype(str) == layer, :], axis=0)
    pos.append(center)
sc.pl.paga(adata, pos=np.array(pos), node_size_scale=2, edge_width_scale=1, fontsize=5, fontoutline=1, ax=ax5)

f.subplots_adjust(hspace=.1, wspace=.1)
save_path = f"../results/downstream/DLPFC/staligner_Sample_all_umap.png"
f.savefig(save_path)


# paste integration
for group in range(len(layer_groups)):
    adata = sc.read_h5ad('../results/DLPFC/paste_integration_center_Sample{}_DLPFC.h5ad'.format(sample_map[group]))
    embedding = reducer.fit_transform(adata.uns['paste_W'])
    adata.obsm["X_umap"] = embedding
    pd.DataFrame(adata.obsm["X_umap"]).to_csv("../results/downstream/DLPFC/paste_integration_center_Sample{}_umap.csv".format(sample_map[group]))

    raw_embedding = reducer.fit_transform(layer_groups[group][1].X)
    layer_groups[group][1].obsm["X_umap"] = raw_embedding
    pd.DataFrame(layer_groups[group][1].obsm["X_umap"]).to_csv("../results/downstream/DLPFC/raw_Sample{}_slice_{}_umap.csv".format(sample_map[group], slice_map[1]))

ratios = [[0.71, 1.1], [1.4, 1.5], [1.8, 1.5]]
for group in range(len(layer_groups)):
    ratio = ratios[group]
    adata = sc.read_h5ad('../results/DLPFC/paste_integration_center_Sample{}_DLPFC.h5ad'.format(sample_map[group]))

    clusters = sc.read_csv("../results/downstream/DLPFC/paste_integration_center_Sample{}_clustering_result.csv".format(sample_map[group])).X
    adata.obs["my_clusters"] = clusters
    raw_clusters = sc.read_csv("../results/downstream/DLPFC/raw_Sample{}_sliceB_clustering_result.csv".format(sample_map[group])).X
    adata.obs["my_clusters_raw"] = raw_clusters
    embedding = sc.read_csv("../results/downstream/DLPFC/paste_integration_center_Sample{}_umap.csv".format(sample_map[group])).X[1:]
    adata.obsm["X_umap"] = embedding
    raw_embedding = sc.read_csv("../results/downstream/DLPFC/raw_Sample{}_slice_{}_umap.csv".format(sample_map[group], slice_map[1])).X[1:]
    adata.obsm["X_umap_raw"] = raw_embedding

    n_spots = embedding.shape[0]
    size = 10000 / n_spots

    sc.pp.neighbors(adata, use_rep='spatial')
    sc.tl.paga(adata, groups='layer_guess_reordered')

    le_layer = preprocessing.LabelEncoder()
    label_layer = le_layer.fit_transform(adata.obs['layer_guess_reordered'])

    order = np.arange(n_spots)
    np.random.shuffle(order)

    f = plt.figure(figsize=(18, 5))

    ax1 = f.add_subplot(1, 4, 1)
    ax1.set_aspect(ratio[0])
    scatter1 = ax1.scatter(raw_embedding[order, 0], raw_embedding[order, 1],
                           s=size, c=adata.obs['my_clusters_raw'][order], cmap='cividis')
    ax1.set_title("Slice B Cluster", fontsize=15)
    ax1.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                    grid_alpha=0)

    l1 = ax1.legend(handles=scatter1.legend_elements()[0],
                    labels=["Cluster %d" % i for i in range(1, 8)],
                    loc="upper left", bbox_to_anchor=(0., 0.),
                    markerscale=1., title_fontsize=10, fontsize=10, frameon=False, ncol=2)
    l1._legend_box.align = "left"

    ax2 = f.add_subplot(1, 4, 2)
    ax2.set_aspect(ratio[1])
    scatter2 = ax2.scatter(embedding[order, 0], embedding[order, 1],
                           s=size, c=adata.obs['my_clusters'][order], cmap='cividis')
    ax2.set_title("Center Cluster", fontsize=15)
    ax2.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                    grid_alpha=0)

    l2 = ax2.legend(handles=scatter2.legend_elements()[0],
                    labels=["Cluster %d" % i for i in range(1, 8)],
                    loc="upper left", bbox_to_anchor=(0., 0.),
                    markerscale=1., title_fontsize=10, fontsize=10, frameon=False, ncol=2)
    l2._legend_box.align = "left"

    ax3 = f.add_subplot(1, 4, 3)
    ax3.set_aspect(ratio[1])
    if group == 1:
        scatter3 = ax3.scatter(embedding[order, 0], embedding[order, 1],
                               s=size, c=label_layer[order], cmap=ListedColormap(
                ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
    else:
        scatter3 = ax3.scatter(embedding[order, 0], embedding[order, 1],
                               s=size, c=label_layer[order], cmap=ListedColormap(
                ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]))
    ax3.set_title("Layer annotation", fontsize=15)
    ax3.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                    grid_alpha=0)

    l3 = ax3.legend(handles=scatter3.legend_elements()[0],
                    labels=sorted(set(adata.obs['layer_guess_reordered'].values)),
                    loc="upper left", bbox_to_anchor=(0., 0.),
                    markerscale=1., title_fontsize=10, fontsize=10, frameon=False, ncol=2)
    l3._legend_box.align = "left"

    ax4 = f.add_subplot(1, 4, 4)
    ax4.set_aspect(ratio[1])
    ax4.set_title("Trajectory", fontsize=15)
    ax4.tick_params(axis='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False,
                    grid_alpha=0)
    ax4.set_xlim(ax3.get_xlim())
    ax4.set_ylim(ax3.get_ylim())

    pos = []
    if group == 1:
        for layer in ["Layer3", "Layer4", "Layer5", "Layer6", "WM"]:
            center = np.mean(embedding[adata.obs['layer_guess_reordered'].values.astype(str) == layer, :], axis=0)
            pos.append(center)
    else:
        for layer in ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"]:
            center = np.mean(embedding[adata.obs['layer_guess_reordered'].values.astype(str) == layer, :], axis=0)
            pos.append(center)
    sc.pl.paga(adata, pos=np.array(pos), node_size_scale=2, edge_width_scale=1, fontsize=5, fontoutline=1, ax=ax4)

    f.subplots_adjust(hspace=.1, wspace=.1)
    save_path = f"../results/downstream/DLPFC/paste_integration_Sample{sample_map[group]}_umap.png"
    f.savefig(save_path)


