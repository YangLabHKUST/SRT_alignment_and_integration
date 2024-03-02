import os
import pickle
import scib
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score

from utils import *

data_dir = '../data/3dst/data/preprocessed/'
save_dir = '../results/RA/'

n_samples = 6
n_slices = [4, 7, 4, 5, 3, 4]
n_clusters = [4, 3, 3, 4, 4, 4]

cls = ['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Collagenous/Adipose', 'Infiltrate;Unknown', 'Unknown']
colors = ['deeppink', 'royalblue', 'blueviolet', 'forestgreen', 'darkgray', 'gray']
sp_cmap = {cls: color for cls, color in zip(cls, colors)}

order_for_clusters = [[1, 3, 2, 0, 5, 4],
                      [0, 2, 1, 4, 5, 3],
                      [0, 3, 1, 4, 2, 5],
                      [1, 4, 2, 0, 3, 5],
                      [1, 4, 2, 0, 3, 5],
                      [1, 3, 2, 0, 4, 5]]

cls_list = [['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Collagenous/Adipose', 'Unknown'],
            ['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Unknown'],
            ['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Infiltrate;Unknown'],
            ['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Collagenous/Adipose', 'Infiltrate;Unknown'],
            ['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Collagenous/Adipose', 'Infiltrate;Unknown'],
            ['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Collagenous/Adipose']]

fig_size = [10, 15, 10, 12, 8.5, 10]
spot_size = [100, 100, 40, 30, 50, 80]

# STAligner sample specific
if not os.path.exists(save_dir + f'STAligner/STAligner_sample_specific_results_dict.pkl'):
    aris = np.zeros((n_samples,), dtype=float)
    b_asws = np.zeros((n_samples,), dtype=float)
    b_pcrs = np.zeros((n_samples,), dtype=float)
    kbets = np.zeros((n_samples,), dtype=float)
    g_conns = np.zeros((n_samples,), dtype=float)

    results_dict = {'ARIs': aris, 'Batch_ASWs': b_asws, 'Batch_PCRs': b_pcrs,
                    'kBETs': kbets, 'Graph_connectivities': g_conns}
    with open(save_dir + f'STAligner/STAligner_sample_specific_results_dict.pkl', 'wb') as file:
        pickle.dump(results_dict, file)

for j in range(n_samples):

    print(f'RA{j+1}')

    slices = []
    for i in range(n_slices[j]):
        adata = ad.read_h5ad(data_dir + f"RA{j+1}/RA{j+1}_slice{i+1}.h5ad")
        adata.obs_names = [x + f'_RA{j+1}_slice{i+1}' for x in adata.obs_names]
        slices.append(adata)
    slice_index_list = list(range(n_slices[j]))
    origin_concat = ad.concat(slices, label='slice_index', keys=slice_index_list)
    adata_concat = origin_concat.copy()

    spots_count = [0]
    n = 0
    for sample in slices:
        num = sample.shape[0]
        n += num
        spots_count.append(n)

    embed = pd.read_csv(save_dir + f'STAligner/STAligner_embed_RA{j+1}.csv', header=None).values
    adata_concat.obsm['latent'] = embed
    gm = GaussianMixture(n_components=n_clusters[j], covariance_type='tied', init_params='kmeans',
                         reg_covar=1e-3, random_state=1234)
    y = gm.fit_predict(adata_concat.obsm['latent'], y=None)
    adata_concat.obs["gm_clusters"] = pd.Series(y, index=adata_concat.obs.index, dtype='category')
    adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata_concat.obs['annotations'],
                                                                          adata_concat.obs["gm_clusters"]),
                                                     index=adata_concat.obs.index, dtype='category')

    order_to_cluster_map = {order: cluster for order, cluster in zip(order_for_clusters[j], cls)}
    adata_concat.obs['matched_clusters'] = list(adata_concat.obs['matched_clusters'].map(order_to_cluster_map))
    for i in range(len(slices)):
        slices[i].obs['matched_clusters'] = adata_concat.obs['matched_clusters'][spots_count[i]: spots_count[i+1]]

    # plot clustering results
    fig, axs = plt.subplots(1, n_slices[j], figsize=(fig_size[j], 2))
    for i in range(len(slices)):
        colors = list(slices[i].obs['matched_clusters'].astype('str').map(sp_cmap))
        axs[i].scatter(slices[i].obsm['spatial'][:, 0], slices[i].obsm['spatial'][:, 1], linewidth=0,
                       s=spot_size[j], marker=".", color=colors)
        axs[i].set_title(f'RA{j + 1} Slice{i + 1}', size=12)
        axs[i].invert_yaxis()
        axs[i].axis('off')

    axs[n_slices[j] - 1].legend(
        handles=[mpatches.Patch(color=sp_cmap[spot_class], label=spot_class) for spot_class in cls_list[j]],
        fontsize=8, title='Clusters', title_fontsize=10, bbox_to_anchor=(1, 1))

    plt.gcf().subplots_adjust(left=0.05, top=None, bottom=None, right=0.8)
    plt.savefig(save_dir + f'STAligner/STAligner_sample_specific_RA{j + 1}_cluster_results.png', dpi=300)

    origin_concat = origin_concat[
        (origin_concat.obs['annotations'] != 'Unknown') & (origin_concat.obs['annotations'] != 'Infiltrate;Unknown')]
    adata_concat = adata_concat[
        (adata_concat.obs['annotations'] != 'Unknown') & (adata_concat.obs['annotations'] != 'Infiltrate;Unknown')]
    adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata_concat.obs['annotations'],
                                                                          adata_concat.obs["gm_clusters"]),
                                                     index=adata_concat.obs.index, dtype='category')

    adata_concat.obs['annotations'] = adata_concat.obs['annotations'].astype(str).astype("category")
    adata_concat.obs['slice_index'] = adata_concat.obs['slice_index'].astype(str).astype("category")
    origin_concat.X = origin_concat.X.astype(float)
    sc.pp.neighbors(adata_concat, use_rep='latent')

    ari = adjusted_rand_score(adata_concat.obs['annotations'], adata_concat.obs['matched_clusters'].tolist())
    b_asw = scib.me.silhouette_batch(adata_concat, batch_key='slice_index', label_key='annotations', embed='latent',
                                     verbose=False)
    b_pcr = scib.me.pcr_comparison(origin_concat, adata_concat, covariate='slice_index', embed='latent')
    kbet = scib.me.kBET(adata_concat, batch_key='slice_index', label_key='annotations', type_='embed', embed='latent')
    g_conn = scib.me.graph_connectivity(adata_concat, label_key='annotations')
    print(ari, b_asw, b_pcr, kbet, g_conn)

    with open(save_dir + f'STAligner/STAligner_sample_specific_results_dict.pkl', 'rb') as file:
        results_dict = pickle.load(file)

    results_dict['ARIs'][j] = ari
    results_dict['Batch_ASWs'][j] = b_asw
    results_dict['Batch_PCRs'][j] = b_pcr
    results_dict['kBETs'][j] = kbet
    results_dict['Graph_connectivities'][j] = g_conn

    with open(save_dir + f'STAligner/STAligner_sample_specific_results_dict.pkl', 'wb') as file:
        pickle.dump(results_dict, file)


# STAligner joint samples
if not os.path.exists(save_dir + f'STAligner/STAligner_joint_samples_results_dict.pkl'):
    aris = np.zeros((n_samples,), dtype=float)
    b_asws = np.zeros((n_samples,), dtype=float)
    b_pcrs = np.zeros((n_samples,), dtype=float)
    kbets = np.zeros((n_samples,), dtype=float)
    g_conns = np.zeros((n_samples,), dtype=float)

    results_dict = {'ARIs': aris, 'Batch_ASWs': b_asws, 'Batch_PCRs': b_pcrs,
                    'kBETs': kbets, 'Graph_connectivities': g_conns}
    with open(save_dir + f'STAligner/STAligner_joint_samples_results_dict.pkl', 'wb') as file:
        pickle.dump(results_dict, file)

class_list = ['Seropositive', 'Seronegative']
for j in range(len(class_list)):

    print(class_list[j])

    samples = []
    spots_count_list = []
    if j == 0:
        sample_index_list = [0, 1, 2]
    else:
        sample_index_list = [3, 4, 5]
    for i in sample_index_list:
        slice_list = []
        slice_index_list = list(range(n_slices[i]))
        for k in range(n_slices[i]):
            adata = ad.read_h5ad(data_dir + f"RA{i+1}/RA{i+1}_slice{k+1}.h5ad")
            adata.obs_names = [x + f'_RA{i+1}_slice{k+1}' for x in adata.obs_names]
            slice_list.append(adata)
        spots_count = [0]
        n = 0
        for sample in slice_list:
            num = sample.shape[0]
            n += num
            spots_count.append(n)
        spots_count_list.append(spots_count)
        samples.append(ad.concat(slice_list, label='slice_index', keys=slice_index_list))

    origin_concat = ad.concat(samples, label='sample_index', keys=sample_index_list)
    adata_concat = origin_concat.copy()

    embed = pd.read_csv(save_dir + f'STAligner/STAligner_embed_{class_list[j]}_RA.csv', header=None).values
    adata_concat.obsm['latent'] = embed
    gm = GaussianMixture(n_components=4, covariance_type='tied', init_params='kmeans',
                         reg_covar=1e-3, random_state=1234)
    y = gm.fit_predict(adata_concat.obsm['latent'], y=None)
    adata_concat.obs["gm_clusters"] = pd.Series(y, index=adata_concat.obs.index, dtype='category')
    origin_concat = origin_concat[
        (origin_concat.obs['annotations'] != 'Unknown') & (origin_concat.obs['annotations'] != 'Infiltrate;Unknown')]
    adata_concat = adata_concat[
        (adata_concat.obs['annotations'] != 'Unknown') & (adata_concat.obs['annotations'] != 'Infiltrate;Unknown')]
    adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata_concat.obs['annotations'],
                                                                          adata_concat.obs["gm_clusters"]),
                                                     index=adata_concat.obs.index, dtype='category')

    for i in sample_index_list:
        adata_concat_sample = adata_concat[adata_concat.obs['sample_index'] == i].copy()
        origin_concat_sample = origin_concat[origin_concat.obs['sample_index'] == i].copy()

        adata_concat_sample.obs['annotations'] = adata_concat_sample.obs['annotations'].astype(str).astype("category")
        adata_concat_sample.obs['slice_index'] = adata_concat_sample.obs['slice_index'].astype(str).astype("category")
        origin_concat_sample.X = origin_concat_sample.X.astype(float)
        sc.pp.neighbors(adata_concat_sample, use_rep='latent')

        ari = adjusted_rand_score(adata_concat_sample.obs['annotations'], adata_concat_sample.obs['matched_clusters'].tolist())
        b_asw = scib.me.silhouette_batch(adata_concat_sample, batch_key='slice_index', label_key='annotations', embed='latent',
                                         verbose=False)
        b_pcr = scib.me.pcr_comparison(origin_concat_sample, adata_concat_sample, covariate='slice_index', embed='latent')
        kbet = scib.me.kBET(adata_concat_sample, batch_key='slice_index', label_key='annotations', type_='embed',
                            embed='latent')
        g_conn = scib.me.graph_connectivity(adata_concat_sample, label_key='annotations')
        print(ari, b_asw, b_pcr, kbet, g_conn)

        with open(save_dir + f'STAligner/STAligner_joint_samples_results_dict.pkl', 'rb') as file:
            results_dict = pickle.load(file)
        results_dict['ARIs'][i] = ari
        results_dict['Batch_ASWs'][i] = b_asw
        results_dict['Batch_PCRs'][i] = b_pcr
        results_dict['kBETs'][i] = kbet
        results_dict['Graph_connectivities'][i] = g_conn
        with open(save_dir + f'STAligner/STAligner_joint_samples_results_dict.pkl', 'wb') as file:
            pickle.dump(results_dict, file)


# PASTE_alignment + STitch3D
if not os.path.exists(save_dir + f'STitch3D/PASTE_alignment+STitch3D_results_dict.pkl'):
    aris = np.zeros((n_samples,), dtype=float)
    b_asws = np.zeros((n_samples,), dtype=float)
    b_pcrs = np.zeros((n_samples,), dtype=float)
    kbets = np.zeros((n_samples,), dtype=float)
    g_conns = np.zeros((n_samples,), dtype=float)

    results_dict = {'ARIs': aris, 'Batch_ASWs': b_asws, 'Batch_PCRs': b_pcrs,
                    'kBETs': kbets, 'Graph_connectivities': g_conns}
    with open(save_dir + f'STitch3D/PASTE_alignment+STitch3D_results_dict.pkl', 'wb') as file:
        pickle.dump(results_dict, file)

for j in range(n_samples):

    print(f'RA{j+1}')

    slices = []
    for i in range(n_slices[j]):
        adata = ad.read_h5ad(data_dir + f"RA{j+1}/RA{j+1}_slice{i+1}.h5ad")
        adata.obs_names = [x + f'_RA{j}_slice{i}' for x in adata.obs_names]
        slices.append(adata)
    slice_index_list = list(range(n_slices[j]))
    origin_concat = ad.concat(slices, label='slice_index', keys=slice_index_list)
    adata_concat = origin_concat.copy()

    spots_count = [0]
    n = 0
    for sample in slices:
        num = sample.shape[0]
        n += num
        spots_count.append(n)

    embed = pd.read_csv(save_dir + f'STitch3D/PASTE_alignment+STitch3D_embed_RA{j+1}.csv', header=None).values
    adata_concat.obsm['latent'] = embed
    gm = GaussianMixture(n_components=n_clusters[j], covariance_type='tied', init_params='kmeans',
                         reg_covar=1e-3, random_state=1234)
    y = gm.fit_predict(adata_concat.obsm['latent'], y=None)
    adata_concat.obs["gm_clusters"] = pd.Series(y, index=adata_concat.obs.index, dtype='category')
    adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata_concat.obs['annotations'],
                                                                          adata_concat.obs["gm_clusters"]),
                                                     index=adata_concat.obs.index, dtype='category')

    order_to_cluster_map = {order: cluster for order, cluster in zip(order_for_clusters[j], cls)}
    adata_concat.obs['matched_clusters'] = list(adata_concat.obs['matched_clusters'].map(order_to_cluster_map))
    for i in range(len(slices)):
        slices[i].obs['matched_clusters'] = adata_concat.obs['matched_clusters'][spots_count[i]: spots_count[i+1]]

    # plot clustering results
    fig, axs = plt.subplots(1, n_slices[j], figsize=(fig_size[j], 2))
    for i in range(len(slices)):
        colors = list(slices[i].obs['matched_clusters'].astype('str').map(sp_cmap))
        axs[i].scatter(slices[i].obsm['spatial'][:, 0], slices[i].obsm['spatial'][:, 1], linewidth=0,
                       s=spot_size[j], marker=".", color=colors)
        axs[i].set_title(f'RA{j + 1} Slice{i + 1}', size=12)
        axs[i].invert_yaxis()
        axs[i].axis('off')

    axs[n_slices[j] - 1].legend(
        handles=[mpatches.Patch(color=sp_cmap[spot_class], label=spot_class) for spot_class in cls_list[j]],
        fontsize=8, title='Clusters', title_fontsize=10, bbox_to_anchor=(1, 1))

    plt.gcf().subplots_adjust(left=0.05, top=None, bottom=None, right=0.8)
    plt.savefig(save_dir + f'STitch3D/PASTE_alignment+STitch3D_RA{j + 1}_cluster_results.png', dpi=300)

    origin_concat = origin_concat[
        (origin_concat.obs['annotations'] != 'Unknown') & (origin_concat.obs['annotations'] != 'Infiltrate;Unknown')]
    adata_concat = adata_concat[
        (adata_concat.obs['annotations'] != 'Unknown') & (adata_concat.obs['annotations'] != 'Infiltrate;Unknown')]
    adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata_concat.obs['annotations'],
                                                                          adata_concat.obs["gm_clusters"]),
                                                     index=adata_concat.obs.index, dtype='category')

    adata_concat.obs['annotations'] = adata_concat.obs['annotations'].astype(str).astype("category")
    adata_concat.obs['slice_index'] = adata_concat.obs['slice_index'].astype(str).astype("category")
    origin_concat.X = origin_concat.X.astype(float)
    sc.pp.neighbors(adata_concat, use_rep='latent')

    ari = adjusted_rand_score(adata_concat.obs['annotations'], adata_concat.obs['matched_clusters'].tolist())
    b_asw = scib.me.silhouette_batch(adata_concat, batch_key='slice_index', label_key='annotations', embed='latent',
                                     verbose=False)
    b_pcr = scib.me.pcr_comparison(origin_concat, adata_concat, covariate='slice_index', embed='latent')
    kbet = scib.me.kBET(adata_concat, batch_key='slice_index', label_key='annotations', type_='embed', embed='latent')
    g_conn = scib.me.graph_connectivity(adata_concat, label_key='annotations')
    print(ari, b_asw, b_pcr, kbet, g_conn)

    with open(save_dir + f'STitch3D/PASTE_alignment+STitch3D_results_dict.pkl', 'rb') as file:
        results_dict = pickle.load(file)

    results_dict['ARIs'][j] = ari
    results_dict['Batch_ASWs'][j] = b_asw
    results_dict['Batch_PCRs'][j] = b_pcr
    results_dict['kBETs'][j] = kbet
    results_dict['Graph_connectivities'][j] = g_conn

    with open(save_dir + f'STitch3D/PASTE_alignment+STitch3D_results_dict.pkl', 'wb') as file:
        pickle.dump(results_dict, file)


# PASTE2 + STitch3D
if not os.path.exists(save_dir + f'STitch3D/PASTE2+STitch3D_results_dict.pkl'):
    aris = np.zeros((n_samples,), dtype=float)
    b_asws = np.zeros((n_samples,), dtype=float)
    b_pcrs = np.zeros((n_samples,), dtype=float)
    kbets = np.zeros((n_samples,), dtype=float)
    g_conns = np.zeros((n_samples,), dtype=float)

    results_dict = {'ARIs': aris, 'Batch_ASWs': b_asws, 'Batch_PCRs': b_pcrs,
                    'kBETs': kbets, 'Graph_connectivities': g_conns}
    with open(save_dir + f'STitch3D/PASTE2+STitch3D_results_dict.pkl', 'wb') as file:
        pickle.dump(results_dict, file)

for j in range(n_samples):

    print(f'RA{j+1}')

    slices = []
    for i in range(n_slices[j]):
        adata = ad.read_h5ad(data_dir + f"RA{j+1}/RA{j+1}_slice{i+1}.h5ad")
        adata.obs_names = [x + f'_RA{j}_slice{i}' for x in adata.obs_names]
        slices.append(adata)
    slice_index_list = list(range(n_slices[j]))
    origin_concat = ad.concat(slices, label='slice_index', keys=slice_index_list)
    adata_concat = origin_concat.copy()

    spots_count = [0]
    n = 0
    for sample in slices:
        num = sample.shape[0]
        n += num
        spots_count.append(n)

    embed = pd.read_csv(save_dir + f'STitch3D/PASTE2+STitch3D_embed_RA{j+1}.csv', header=None).values
    adata_concat.obsm['latent'] = embed
    gm = GaussianMixture(n_components=n_clusters[j], covariance_type='tied', init_params='kmeans',
                         reg_covar=1e-3, random_state=1234)
    y = gm.fit_predict(adata_concat.obsm['latent'], y=None)
    adata_concat.obs["gm_clusters"] = pd.Series(y, index=adata_concat.obs.index, dtype='category')
    adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata_concat.obs['annotations'],
                                                                          adata_concat.obs["gm_clusters"]),
                                                     index=adata_concat.obs.index, dtype='category')

    order_to_cluster_map = {order: cluster for order, cluster in zip(order_for_clusters[j], cls)}
    adata_concat.obs['matched_clusters'] = list(adata_concat.obs['matched_clusters'].map(order_to_cluster_map))
    for i in range(len(slices)):
        slices[i].obs['matched_clusters'] = adata_concat.obs['matched_clusters'][spots_count[i]: spots_count[i+1]]

    # plot clustering results
    fig, axs = plt.subplots(1, n_slices[j], figsize=(fig_size[j], 2))
    for i in range(len(slices)):
        colors = list(slices[i].obs['matched_clusters'].astype('str').map(sp_cmap))
        axs[i].scatter(slices[i].obsm['spatial'][:, 0], slices[i].obsm['spatial'][:, 1], linewidth=0,
                       s=spot_size[j], marker=".", color=colors)
        axs[i].set_title(f'RA{j + 1} Slice{i + 1}', size=12)
        axs[i].invert_yaxis()
        axs[i].axis('off')

    axs[n_slices[j] - 1].legend(
        handles=[mpatches.Patch(color=sp_cmap[spot_class], label=spot_class) for spot_class in cls_list[j]],
        fontsize=8, title='Clusters', title_fontsize=10, bbox_to_anchor=(1, 1))

    plt.gcf().subplots_adjust(left=0.05, top=None, bottom=None, right=0.8)
    plt.savefig(save_dir + f'STitch3D/PASTE2+STitch3D_RA{j + 1}_cluster_results.png', dpi=300)

    origin_concat = origin_concat[
        (origin_concat.obs['annotations'] != 'Unknown') & (origin_concat.obs['annotations'] != 'Infiltrate;Unknown')]
    adata_concat = adata_concat[
        (adata_concat.obs['annotations'] != 'Unknown') & (adata_concat.obs['annotations'] != 'Infiltrate;Unknown')]
    adata_concat.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata_concat.obs['annotations'],
                                                                          adata_concat.obs["gm_clusters"]),
                                                     index=adata_concat.obs.index, dtype='category')

    adata_concat.obs['annotations'] = adata_concat.obs['annotations'].astype(str).astype("category")
    adata_concat.obs['slice_index'] = adata_concat.obs['slice_index'].astype(str).astype("category")
    origin_concat.X = origin_concat.X.astype(float)
    sc.pp.neighbors(adata_concat, use_rep='latent')

    ari = adjusted_rand_score(adata_concat.obs['annotations'], adata_concat.obs['matched_clusters'].tolist())
    b_asw = scib.me.silhouette_batch(adata_concat, batch_key='slice_index', label_key='annotations', embed='latent',
                                     verbose=False)
    b_pcr = scib.me.pcr_comparison(origin_concat, adata_concat, covariate='slice_index', embed='latent')
    kbet = scib.me.kBET(adata_concat, batch_key='slice_index', label_key='annotations', type_='embed', embed='latent')
    g_conn = scib.me.graph_connectivity(adata_concat, label_key='annotations')
    print(ari, b_asw, b_pcr, kbet, g_conn)

    with open(save_dir + f'STitch3D/PASTE2+STitch3D_results_dict.pkl', 'rb') as file:
        results_dict = pickle.load(file)

    results_dict['ARIs'][j] = ari
    results_dict['Batch_ASWs'][j] = b_asw
    results_dict['Batch_PCRs'][j] = b_pcr
    results_dict['kBETs'][j] = kbet
    results_dict['Graph_connectivities'][j] = g_conn

    with open(save_dir + f'STitch3D/PASTE2+STitch3D_results_dict.pkl', 'wb') as file:
        pickle.dump(results_dict, file)


# PASTE_integration
if not os.path.exists(save_dir + f'PASTE_integration/PASTE_integration_results_dict.pkl'):
    aris = np.zeros((n_samples,), dtype=float)
    results_dict = {'ARIs': aris}
    with open(save_dir + f'PASTE_integration/PASTE_integration_results_dict.pkl', 'wb') as file:
        pickle.dump(results_dict, file)

for j in range(n_samples):

    print(f'RA{j + 1}')

    adata = ad.read_h5ad(data_dir + f"RA{j + 1}/RA{j + 1}_slice2.h5ad")

    embed = pd.read_csv(save_dir + f'PASTE_integration/PASTE_integration_embed_RA{j+1}.csv', header=None).values
    adata.obsm['latent'] = embed
    gm = GaussianMixture(n_components=n_clusters[j], covariance_type='tied', init_params='kmeans',
                         reg_covar=1e-3, random_state=1234)
    y = gm.fit_predict(adata.obsm['latent'], y=None)
    adata.obs["gm_clusters"] = pd.Series(y, index=adata.obs.index, dtype='category')
    adata = adata[(adata.obs['annotations'] != 'Unknown') & (adata.obs['annotations'] != 'Infiltrate;Unknown')]
    adata.obs['matched_clusters'] = pd.Series(match_cluster_labels(adata.obs['annotations'], adata.obs["gm_clusters"]),
                                              index=adata.obs.index, dtype='category')

    ari = adjusted_rand_score(adata.obs['annotations'], adata.obs['matched_clusters'].tolist())
    print(ari)

    with open(save_dir + f'PASTE_integration/PASTE_integration_results_dict.pkl', 'rb') as file:
        results_dict = pickle.load(file)
    results_dict['ARIs'][j] = ari
    with open(save_dir + f'PASTE_integration/PASTE_integration_results_dict.pkl', 'wb') as file:
        pickle.dump(results_dict, file)
