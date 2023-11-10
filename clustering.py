from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.patches as mpatches
import sklearn
from src.paste2.model_selection import *
from src.paste2.helper import *
from utils import match_cluster_labels
import warnings
warnings.filterwarnings("ignore")

np.random.seed(1234)
slice_map = {0:'A',1:'B',2:'C',3:'D'}
sample_map = {0:'A',1:'B',2:'C'}


def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # sc.pp.filter_cells(adata, min_counts=100)
    # sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    # sc.pp.filter_genes(adata, min_cells=10)

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    adata.X = adata.X.toarray()
    return adata


# using DLPFC dataset as an example
sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
adatas = {sample:sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
sample_groups = [["151507", "151508", "151509","151510"], [ "151669", "151670","151671", "151672"], [ "151673","151674", "151675", "151676"]]
layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]
layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]


# raw cluster
for i in range(len(layer_groups)):
    print('sample'+sample_map[i])

    if i == 1:
        gm = GaussianMixture(n_components=5, covariance_type='tied', init_params='kmeans', reg_covar=1e-3)
    else:
        gm = GaussianMixture(n_components=7, covariance_type='tied', init_params='kmeans', reg_covar=1e-3)

    for j in range(len(layer_groups[i])):
        adata = layer_groups[i][j].copy()
        # process_data(adata, n_top_genes=6000)
        y = gm.fit_predict(adata.X, y=None)
        adata.obs["GM"] = y
        if i == 1:
            adata.obs['my_clusters'] = pd.Series(3 + match_cluster_labels(
                adata.obs['layer_guess_reordered'], adata.obs["GM"]),
                                                              index=adata.obs.index, dtype='category')
        else:
            adata.obs['my_clusters'] = pd.Series(1 + match_cluster_labels(
                adata.obs['layer_guess_reordered'],adata.obs["GM"]),
                                                              index=adata.obs.index,dtype='category')
        adata.obs["my_clusters"].to_csv(f"../results/downstream/DLPFC/raw_Sample{sample_map[i]}_slice{slice_map[j]}_clustering_result.csv")

spots_count = [[0], [0], [0]]
for i in range(len(sample_groups)):
    n = 0
    for sample in sample_groups[i]:
        num = adatas[sample].shape[0]
        n += num
        spots_count[i].append(n)

raw = []
for i in range(len(sample_groups)):

    adata1 = layer_groups[i][0].copy()
    clusters1 = sc.read_csv(f"../results/downstream/DLPFC/raw_Sample{sample_map[i]}_sliceA_clustering_result.csv").X
    adata1.obs["my_clusters"] = clusters1
    adata2 = layer_groups[i][1].copy()
    clusters2 = sc.read_csv(f"../results/downstream/DLPFC/raw_Sample{sample_map[i]}_sliceB_clustering_result.csv").X
    adata2.obs["my_clusters"] = clusters2
    adata3 = layer_groups[i][2].copy()
    clusters3 = sc.read_csv(f"../results/downstream/DLPFC/raw_Sample{sample_map[i]}_sliceC_clustering_result.csv").X
    adata3.obs["my_clusters"] = clusters3
    adata4 = layer_groups[i][3].copy()
    clusters4 = sc.read_csv(f"../results/downstream/DLPFC/raw_Sample{sample_map[i]}_sliceD_clustering_result.csv").X
    adata4.obs["my_clusters"] = clusters4
    clusters = [clusters1, clusters2, clusters3, clusters4]

    adata = ad.concat([adata1, adata2, adata3, adata4])
    raw.append(adata)
    for j in range(len(layer_groups[i])):
        layer_groups[i][j].obs['my_clusters'] = clusters[j]

fig, axs = plt.subplots(3, 4, figsize=(15, 11.5))
fig.suptitle('Raw data clustering', fontsize=20)
for j in range(len(layer_groups)):
    ari_all = sklearn.metrics.adjusted_rand_score(raw[j].obs['layer_guess_reordered'],
                                                  raw[j].obs['my_clusters'])
    axs[j,0].text(-0.1, 0.5, 'Sample ' + sample_map[j] + ' ARI=' + "{:.2f}".format(ari_all),fontsize=12,rotation='vertical',transform = axs[j,0].transAxes,verticalalignment='center')
    for i in range(len(layer_groups[j])):
        ari = sklearn.metrics.adjusted_rand_score(layer_groups[j][i].obs['layer_guess_reordered'],
                                                  layer_groups[j][i].obs['my_clusters'])
        adata = layer_groups[j][i]
        colors = [sns.color_palette()[int(label-1)] for label in adata.obs['my_clusters']]
        axs[j,i].scatter(layer_groups[j][i].obsm['spatial'][:,0],layer_groups[j][i].obsm['spatial'][:,1],linewidth=0,s=20, marker=".",color=colors)
        axs[j,i].set_title('Slice ' + slice_map[i] + ' ARI=' + "{:.2f}".format(ari), size=12)
        axs[j,i].invert_yaxis()
        axs[j,i].axis('off')
    axs[j,3].legend(handles=[mpatches.Patch(color=layer_to_color_map[adata.obs['layer_guess_reordered'].cat.categories[i]], label=adata.obs['layer_guess_reordered'].cat.categories[i]) for i in range(len(adata.obs['layer_guess_reordered'].cat.categories))],fontsize=10,title='Cortex layer',title_fontsize=12,bbox_to_anchor=(1, 1))
save_path = "../results/downstream/DLPFC/raw_clustering.png"
plt.savefig(save_path)
plt.show()


# paste integration
paste_A = sc.read_h5ad('../results/DLPFC/paste_integration_center_SampleA_DLPFC.h5ad')
paste_B = sc.read_h5ad('../results/DLPFC/paste_integration_center_SampleB_DLPFC.h5ad')
paste_C = sc.read_h5ad('../results/DLPFC/paste_integration_center_SampleC_DLPFC.h5ad')
paste_integration = [paste_A, paste_B, paste_C]
for i in range(len(paste_integration)):
    print('sample' + sample_map[i])
    if i == 1:
        gm = GaussianMixture(n_components=5, covariance_type='tied', init_params='kmeans', reg_covar=1e-3)
    else:
        gm = GaussianMixture(n_components=7, covariance_type='tied', init_params='kmeans', reg_covar=1e-3)
    W = sc.AnnData(paste_integration[i].uns['paste_W'])
    sc.pp.normalize_total(W, inplace=True)
    sc.pp.log1p(W)
    y = gm.fit_predict(W.X, y=None)
    paste_integration[i].obs["GM"] = y
    if i == 1:
        paste_integration[i].obs['my_clusters'] = pd.Series(3 + match_cluster_labels(
            paste_integration[i].obs['layer_guess_reordered'], paste_integration[i].obs["GM"]),
                                                          index=paste_integration[i].obs.index, dtype='category')
    else:
        paste_integration[i].obs['my_clusters'] = pd.Series(1 + match_cluster_labels(
            paste_integration[i].obs['layer_guess_reordered'], paste_integration[i].obs["GM"]),
                                                          index=paste_integration[i].obs.index, dtype='category')
    paste_integration[i].obs["my_clusters"].to_csv(f"../results/downstream/DLPFC/paste_integration_center"
                                                   f"_Sample{sample_map[i]}_clustering_result.csv")

for i in range(len(paste_integration)):
    paste_integration[i].obs["my_clusters"] = sc.read_csv(f"../results/downstream/DLPFC/paste_integration_center"
                                                          f"_Sample{sample_map[i]}_clustering_result.csv").X

for i in range(len(paste_integration)):
    plt.figure(figsize=(7, 7))
    ari = sklearn.metrics.adjusted_rand_score(paste_integration[i].obs['layer_guess_reordered'],
                                              paste_integration[i].obs['my_clusters'])
    adata = paste_integration[i]
    colors = [sns.color_palette()[int(label - 1)] for label in adata.obs['my_clusters']]
    plt.scatter(paste_integration[i].obsm['spatial'][:, 0], paste_integration[i].obsm['spatial'][:, 1],
                linewidth=0, s=70, marker=".", color=colors)
    plt.title('Sample ' + sample_map[i] + ' center ARI=' + "{:.2f}".format(ari), size=12)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.legend(handles=[mpatches.Patch(color=layer_to_color_map[adata.obs['layer_guess_reordered'].cat.categories[i]],
               label=adata.obs['layer_guess_reordered'].cat.categories[i]) for i in
                        range(len(adata.obs['layer_guess_reordered'].cat.categories))], fontsize=6,
               title='Cortex layer', title_fontsize=6, bbox_to_anchor=(1, 1))
    save_path = f"../results/downstream/DLPFC/paste_integration_center_Sample{sample_map[i]}_clustering.png"
    plt.savefig(save_path)
    plt.show()


# staligner sample specific
sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
adatas = {sample:sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
sample_groups = [["151507", "151508", "151509","151510"], [ "151669", "151670","151671", "151672"], [ "151673","151674", "151675", "151676"]]
layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]
layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]

for i in range(len(layer_groups)):
    print('sample' + sample_map[i])
    adata = sc.read_h5ad('../results/DLPFC/staligner_Sample{}_DLPFC.h5ad'.format(sample_map[i]))
    adata.obs.rename(columns={'Ground Truth': 'layer_guess_reordered'}, inplace=True)

    if i == 1:
        gm = GaussianMixture(n_components=5, covariance_type='tied', init_params='kmeans', reg_covar=1e-3)
    else:
        gm = GaussianMixture(n_components=7, covariance_type='tied', init_params='kmeans', reg_covar=1e-3)

    y = gm.fit_predict(adata.obsm['STAligner'], y=None)
    adata.obs["GM"] = y
    if i == 1:
        adata.obs['my_clusters'] = pd.Series(3 + match_cluster_labels(adata.obs['layer_guess_reordered'], adata.obs["GM"]),
                                             index=adata.obs.index, dtype='category')
    else:
        adata.obs['my_clusters'] = pd.Series(1 + match_cluster_labels(adata.obs['layer_guess_reordered'], adata.obs["GM"]),
                                             index=adata.obs.index,dtype='category')
    adata.obs["my_clusters"].to_csv(f"../results/downstream/DLPFC/staligner_Sample{sample_map[i]}_clustering_result.csv")

staligner_results = []
for i in range(len(sample_groups)):
    adata = sc.read_h5ad('../results/DLPFC/staligner_Sample{}_DLPFC.h5ad'.format(sample_map[i]))
    adata.obs.rename(columns={'Ground Truth': 'layer_guess_reordered'}, inplace=True)
    clusters = sc.read_csv(f"../results/downstream/DLPFC/staligner_Sample{sample_map[i]}_clustering_result.csv").X
    adata.obs["my_clusters"] = clusters
    staligner_results.append(adata)
    for j in range(len(layer_groups[i])):
        layer_groups[i][j].obs['my_clusters'] = clusters[spots_count[i][j]: spots_count[i][j+1]]

fig, axs = plt.subplots(3, 4, figsize=(15, 11.5))
fig.suptitle('STAligner sample specific data clustering', fontsize=20)
for j in range(len(layer_groups)):
    ari_all = sklearn.metrics.adjusted_rand_score(staligner_results[j].obs['layer_guess_reordered'],
                                                  staligner_results[j].obs['my_clusters'])
    axs[j,0].text(-0.1, 0.5, 'Sample ' + sample_map[j] + ' ARI=' + "{:.2f}".format(ari_all),fontsize=12,rotation='vertical',transform = axs[j,0].transAxes,verticalalignment='center')
    for i in range(len(layer_groups[j])):
        ari = sklearn.metrics.adjusted_rand_score(layer_groups[j][i].obs['layer_guess_reordered'],
                                                  layer_groups[j][i].obs['my_clusters'])
        adata = layer_groups[j][i]
        colors = [sns.color_palette()[int(label-1)] for label in adata.obs['my_clusters']]
        # colors = list(adata.obs['layer_guess_reordered'].astype('str').map(layer_to_color_map))
        axs[j,i].scatter(layer_groups[j][i].obsm['spatial'][:,0],layer_groups[j][i].obsm['spatial'][:,1],linewidth=0,s=20, marker=".",color=colors)
        axs[j,i].set_title('Slice ' + slice_map[i] + ' ARI=' + "{:.2f}".format(ari), size=12)
        axs[j,i].invert_yaxis()
        axs[j,i].axis('off')
    axs[j,3].legend(handles=[mpatches.Patch(color=layer_to_color_map[adata.obs['layer_guess_reordered'].cat.categories[i]], label=adata.obs['layer_guess_reordered'].cat.categories[i]) for i in range(len(adata.obs['layer_guess_reordered'].cat.categories))],fontsize=10,title='Cortex layer',title_fontsize=12,bbox_to_anchor=(1, 1))
save_path = "../results/downstream/DLPFC/staligner_sample_specific_clustering.png"
plt.savefig(save_path)
plt.show()


# staligner joint samples
print('start clustering STAligner results of all data jointly')
staligner_all = sc.read_h5ad('../results/DLPFC/staligner_Sample_all_DLPFC.h5ad')
staligner_all.obs.rename(columns={'Ground Truth': 'layer_guess_reordered'}, inplace=True)
# sc.pp.normalize_total(staligner_all, inplace=True)
# sc.pp.log1p(staligner_all)
gm = GaussianMixture(n_components=7, covariance_type='tied', init_params='kmeans', reg_covar=1e-3)
y = gm.fit_predict(staligner_all.obsm['STAligner'], y=None)
staligner_all.obs["GM"] = y
staligner_all.obs['my_clusters'] = pd.Series(1 + match_cluster_labels(staligner_all.obs['layer_guess_reordered'], staligner_all.obs["GM"]),
                                             index=staligner_all.obs.index, dtype='category')
staligner_all.obs["my_clusters"].to_csv("../results/downstream/DLPFC/staligner_Sample_all_clustering_result.csv")

sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
adatas = {sample:sc.read_h5ad('../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
sample_groups = [["151507", "151508", "151509","151510"], [ "151669", "151670","151671", "151672"], [ "151673","151674", "151675", "151676"]]
layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]
layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]

spots_count_all = [0]
n = 0
for sample in sample_list:
    num = adatas[sample].shape[0]
    n += num
    spots_count_all.append(n)

staligner_all = sc.read_h5ad('../results/DLPFC/staligner_Sample_all_DLPFC.h5ad')
staligner_all.obs.rename(columns={'Ground Truth': 'layer_guess_reordered'}, inplace=True)
clusters = sc.read_csv("../results/downstream/DLPFC/staligner_Sample_all_clustering_result.csv").X
staligner_all.obs['my_clusters'] = clusters
for i in range(len(layer_groups)):
    for j in range(len(layer_groups[i])):
        layer_groups[i][j].obs['my_clusters'] = clusters[spots_count_all[i * 4 + j]: spots_count_all[i * 4 + j + 1]]

fig, axs = plt.subplots(3, 4, figsize=(15, 11.5))
fig.suptitle('STAligner joint samples data clustering', fontsize=20)
for j in range(len(layer_groups)):
    ari_all = sklearn.metrics.adjusted_rand_score(staligner_all.obs['layer_guess_reordered'][spots_count_all[j*4]: spots_count_all[j*4+4]],
                                                  staligner_all.obs['my_clusters'][spots_count_all[j*4]: spots_count_all[j*4+4]])
    axs[j,0].text(-0.1, 0.5, 'Sample ' + sample_map[j] + ' ARI=' + "{:.2f}".format(ari_all),fontsize=12,rotation='vertical',transform = axs[j,0].transAxes,verticalalignment='center')
    for i in range(len(layer_groups[j])):
        ari = sklearn.metrics.adjusted_rand_score(layer_groups[j][i].obs['layer_guess_reordered'],
                                                  layer_groups[j][i].obs['my_clusters'])
        adata = layer_groups[j][i]
        colors = [sns.color_palette()[int(label-1)] for label in adata.obs['my_clusters']]
        axs[j,i].scatter(layer_groups[j][i].obsm['spatial'][:,0],layer_groups[j][i].obsm['spatial'][:,1],linewidth=0,s=20, marker=".",color=colors)
        axs[j,i].set_title('Slice ' + slice_map[i] + ' ARI=' + "{:.2f}".format(ari), size=12)
        axs[j,i].invert_yaxis()
        axs[j,i].axis('off')
    axs[j,3].legend(handles=[mpatches.Patch(color=layer_to_color_map[layer_groups[0][0].obs['layer_guess_reordered'].cat.categories[i]], label=layer_groups[0][0].obs['layer_guess_reordered'].cat.categories[i]) for i in range(len(layer_groups[0][0].obs['layer_guess_reordered'].cat.categories))],fontsize=10,title='Cortex layer',title_fontsize=12,bbox_to_anchor=(1, 1))
save_path = "../results/downstream/DLPFC/staligner_all_samples_clustering.png"
plt.savefig(save_path)
plt.show()
