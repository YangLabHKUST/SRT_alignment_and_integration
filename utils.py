import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ot
import scipy
from scipy.spatial import distance
import anndata as ad
import scanpy as sc
import seaborn as sns

from STAligner.ST_utils import best_fit_transform
from STAligner.mnn_utils import *
from paste.helper import to_dense_array


def ICP_align(adata_concat, adata_target, adata_ref, slice_target, slice_ref, k=1, plot_align=False):
    ### find MNN pairs in the landmark domain with knn=1
    adata_slice1 = adata_target
    adata_slice2 = adata_ref

    batch_pair = adata_concat[adata_concat.obs['batch_name'].isin([slice_target, slice_ref])]
    mnn_dict = create_dictionary_mnn(batch_pair, use_rep='STAligner', batch_name='batch_name', k=k, iter_comb=None,
                                     verbose=0)
    # print(len(list(mnn_dict[list(mnn_dict.keys())[0]].keys()))/2)
    adata_1 = batch_pair[batch_pair.obs['batch_name'] == slice_target]
    adata_2 = batch_pair[batch_pair.obs['batch_name'] == slice_ref]

    anchor_list = []
    positive_list = []
    for batch_pair_name in mnn_dict.keys():
        for anchor in mnn_dict[batch_pair_name].keys():
            positive_spot = mnn_dict[batch_pair_name][anchor][0]
            ### anchor should only in the ref slice, pos only in the target slice
            if anchor in adata_1.obs_names and positive_spot in adata_2.obs_names:
                anchor_list.append(anchor)
                positive_list.append(positive_spot)

    batch_as_dict = dict(zip(list(adata_concat.obs_names), range(0, adata_concat.shape[0])))
    anchor_ind = list(map(lambda _: batch_as_dict[_], anchor_list))
    positive_ind = list(map(lambda _: batch_as_dict[_], positive_list))
    anchor_arr = adata_concat.obsm['STAligner'][anchor_ind,]
    positive_arr = adata_concat.obsm['STAligner'][positive_ind,]
    dist_list = [np.sqrt(np.sum(np.square(anchor_arr[ii, :] - positive_arr[ii, :]))) for ii in
                 range(anchor_arr.shape[0])]

    key_points_src = np.array(anchor_list)[dist_list < np.percentile(dist_list, 50)]  ## remove remote outliers
    key_points_dst = np.array(positive_list)[dist_list < np.percentile(dist_list, 50)]
    # print(len(anchor_list), len(key_points_src))

    coor_src = adata_slice1.obsm["spatial"]  ## to_be_aligned
    coor_dst = adata_slice2.obsm["spatial"]  ## reference_points

    ## index number
    MNN_ind_src = [list(adata_1.obs_names).index(key_points_src[ii]) for ii in range(len(key_points_src))]
    MNN_ind_dst = [list(adata_2.obs_names).index(key_points_dst[ii]) for ii in range(len(key_points_dst))]

    ####### ICP alignment
    init_pose = None
    max_iterations = 100
    tolerance = 0.001

    coor_used = coor_src  ## Batch_list[1][Batch_list[1].obs['annotation']==2].obsm["spatial"]
    coor_all = adata_target.obsm["spatial"].copy()
    coor_used = np.concatenate([coor_used, np.expand_dims(np.ones(coor_used.shape[0]), axis=1)], axis=1).T
    coor_all = np.concatenate([coor_all, np.expand_dims(np.ones(coor_all.shape[0]), axis=1)], axis=1).T
    A = coor_src  ## to_be_aligned
    B = coor_dst  ## reference_points

    m = A.shape[1]  # get number of dimensions

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = 0

    for ii in range(max_iterations + 1):
        p1 = src[:m, MNN_ind_src].T
        p2 = dst[:m, MNN_ind_dst].T
        T, _, _ = best_fit_transform(src[:m, MNN_ind_src].T,
                                     dst[:m, MNN_ind_dst].T)  ## compute the transformation matrix based on MNNs
        import math
        distances = np.mean([math.sqrt(((p1[kk, 0] - p2[kk, 0]) ** 2) + ((p1[kk, 1] - p2[kk, 1]) ** 2))
                             for kk in range(len(p1))])

        # update the current source
        src = np.dot(T, src)
        coor_used = np.dot(T, coor_used)
        coor_all = np.dot(T, coor_all)

        # check error
        mean_error = np.mean(distances)
        # print(mean_error)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    aligned_points = coor_used.T  # MNNs in the landmark_domain
    aligned_points_all = coor_all.T  # all points in the slice

    if plot_align:
        import matplotlib.pyplot as plt
        plt.rcParams["figure.figsize"] = (3, 3)
        fig, ax = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'wspace': 0.5, 'hspace': 0.1})
        ax[0].scatter(adata_slice2.obsm["spatial"][:, 0], adata_slice2.obsm["spatial"][:, 1],
                      c="blue", cmap=plt.cm.binary_r, s=1)
        ax[0].set_title('Reference ' + slice_ref, size=14)
        ax[1].scatter(aligned_points[:, 0], aligned_points[:, 1],
                      c="blue", cmap=plt.cm.binary_r, s=1)
        ax[1].set_title('Target ' + slice_target, size=14)

        plt.axis("equal")
        # plt.axis("off")
        plt.show()

    # adata_target.obsm["spatial"] = aligned_points_all[:,:2]
    return aligned_points_all[:, :2]


def ratio_and_accuracy(adata, adata_target, adata_ref, slice_target, slice_ref, k=1, use_rep='STAligner', ratio=0.8, kmax=51):
    r = 0
    while r < ratio and k <= kmax:
        # print(k)
        batch_pair = adata[adata.obs['batch_name'].isin([slice_target, slice_ref])]
        mnn_dict = create_dictionary_mnn(batch_pair, use_rep=use_rep, batch_name='batch_name', k=k, iter_comb=None,
                                         verbose=0)
        ref_amount = len(adata_ref.obs_names)
        ref_spots = 0
        # target_amount = len(adata_target.obs_names)
        # target_spots = 0
        all = 0
        correct = 0
        for key in mnn_dict[list(mnn_dict.keys())[0]].keys():
            if key in adata_ref.obs_names:
                ref_spots += 1
                pos1 = adata_ref.obs_names.tolist().index(key)
                for i in mnn_dict[list(mnn_dict.keys())[0]][key]:
                    pos2 = adata_target.obs_names.tolist().index(i)
                    all += 1
                    if adata_ref.obs['Ground Truth'][pos1] == adata_target.obs['Ground Truth'][pos2]:
                        correct += 1
            # elif key in adata_target.obs_names:
            #     target_spots += 1
            #     pos1 = adata_target.obs_names.tolist().index(key)
            #     for i in mnn_dict[list(mnn_dict.keys())[0]][key]:
            #         pos2 = adata_ref.obs_names.tolist().index(i)
            #         all += 1
            #         if adata_target.obs['Ground Truth'][pos1] == adata_ref.obs['Ground Truth'][pos2]:
            #             correct += 1
            # else:
            #     print('no spot named' + key)
        accu = correct / all
        r = round(ref_spots/ref_amount, 2)
        if r < ratio:
            k += 5
        # return [[ref_amount, target_amount, ref_spots, target_spots, all], accu]
    return [round(ref_spots/ref_amount, 2), round(accu, 2)]


def match_cluster_labels(true_labels, est_labels):
    true_labels_arr = np.array(list(true_labels))
    est_labels_arr = np.array(list(est_labels))
    org_cat = list(np.sort(list(pd.unique(true_labels))))
    est_cat = list(np.sort(list(pd.unique(est_labels))))
    B = nx.Graph()
    B.add_nodes_from([i + 1 for i in range(len(org_cat))], bipartite=0)
    B.add_nodes_from([-j - 1 for j in range(len(est_cat))], bipartite=1)
    for i in range(len(org_cat)):
        for j in range(len(est_cat)):
            weight = np.sum((true_labels_arr == org_cat[i]) * (est_labels_arr == est_cat[j]))
            B.add_edge(i + 1, -j - 1, weight=-weight)
    match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)
    #     match = minimum_weight_full_matching(B)
    if len(org_cat) >= len(est_cat):
        return np.array([match[-est_cat.index(c) - 1] - 1 for c in est_labels_arr])
    else:
        unmatched = [c for c in est_cat if not (-est_cat.index(c) - 1) in match.keys()]
        l = []
        for c in est_labels_arr:
            if (-est_cat.index(c) - 1) in match:
                l.append(match[-est_cat.index(c) - 1] - 1)
            else:
                l.append(len(org_cat) + unmatched.index(c))
        return np.array(l)


def nn_accuracy(adata1, adata2):
    nearest_neighbors = []
    for i in range(len(adata1.obsm['spatial'])):
        point1 = adata1.obsm['spatial'][i]
        min_distance = float('inf')
        nearest_neighbor = None
        for j in range(len(adata2.obsm['spatial'])):
            point2 = adata2.obsm['spatial'][j]
            dist = distance.euclidean(point1, point2)
            if dist < min_distance:
                min_distance = dist
                nearest_neighbor = j

        nearest_neighbors.append(nearest_neighbor)

    correct = 0
    for i in range(len(nearest_neighbors)):
        if adata1.obs['Ground Truth'][i] == adata2.obs['Ground Truth'][nearest_neighbors[i]]:
            correct += 1

    return round(correct/len(nearest_neighbors), 2)


def partial_cut(adata, percentage, is_left=True):
    if is_left:
        x_threshold = np.percentile(adata.obsm['spatial'][:, 0], percentage * 100)
        selected_indices = np.where(adata.obsm['spatial'][:, 0] <= x_threshold)[0]
    else:
        x_threshold = np.percentile(adata.obsm['spatial'][:, 0], 100 - percentage * 100)
        selected_indices = np.where(adata.obsm['spatial'][:, 0] >= x_threshold)[0]
    filtered_data = adata[selected_indices]
    filtered_anndata = ad.AnnData(
        X=filtered_data.X,
        obs=filtered_data.obs,
        var=filtered_data.var,
        uns=filtered_data.uns,
    )
    filtered_anndata.obsm['spatial'] = adata.obsm['spatial'][selected_indices]
    return filtered_anndata


def max_accuracy(labels1,labels2):
    w = min(1/len(labels1),1/len(labels2))
    cats = set(pd.unique(labels1)).union(set(pd.unique(labels1)))  ## labels2
    return sum([w * min(sum(labels1==c),sum(labels2==c)) for c in cats])


def mapping_accuracy(labels1,labels2,pi):
    mapping_dict = {'Layer1':1, 'Layer2':2, 'Layer3':3, 'Layer4':4, 'Layer5':5, 'Layer6':6, 'WM':7}
    return np.sum(pi*(scipy.spatial.distance_matrix(np.matrix(labels1.map(mapping_dict) ).T,np.matrix(labels2.map(mapping_dict)).T)==0))


def max_accuracy_mapping(labels1,labels2):
    n1,n2=len(labels1),len(labels2)
    mapping_dict = {'Layer1':1, 'Layer2':2, 'Layer3':3, 'Layer4':4, 'Layer5':5, 'Layer6':6, 'WM':7}
    dist = np.array(scipy.spatial.distance_matrix(np.matrix(labels1.map(mapping_dict)).T,np.matrix(labels2.map(mapping_dict)).T)!=0,dtype=float)
    pi = ot.emd(np.ones(n1)/n1, np.ones(n2)/n2, dist)
    return pi


def get_scatter_contours(layer, labels, interest=['WM']):
    idx = np.array(range(len(labels)))[(labels.isin(interest)).to_numpy()]
    idx_not = np.array(range(len(labels)))[(labels.isin(set(labels.cat.categories).difference(interest))).to_numpy()]
    dist = scipy.spatial.distance_matrix(layer.obsm['spatial'], layer.obsm['spatial'])
    min_dist = np.min(dist[dist > 0])
    eps = 0.01
    edges = np.zeros(dist.shape)
    edges[dist > 0] = (dist[dist > 0] - min_dist) ** 2 < eps
    border = list(filter(lambda x: np.sum(edges[x, idx_not] > 0), idx))
    j = np.argmin(layer.obsm['spatial'][border, 0])
    contours, left = [[border[j]]], set(border).difference(set([border[j]]))
    for i in range(1, len(border)):
        last = contours[-1][-1]
        neighbors = set(left).intersection(np.where((dist[last, :] - min_dist) ** 2 < eps)[0])
        if len(neighbors) > 0:
            j = neighbors.pop()
            contours[-1].append(j)
        else:
            l = list(left)
            j = l[np.argmin(layer.obsm['spatial'][l, 0])]
            contours.append([j])
        left = left.difference(set([j]))
    return contours


def plot_2d_expression(layer, gene, adatas, sample_list, name="", title='', vmin=None, vmax=None, layer_idx=None,
                       norm=False, draw_contours=True):
    cmap = sns.color_palette("rocket_r",
                             as_cmap=True)  # sns.color_palette("magma", as_cmap=True) #sns.cubehelix_palette(as_cmap=True)
    fig = plt.figure(figsize=(10, 10))
    v = to_dense_array(layer[:, gene].X).copy().ravel() + 1
    if norm: v = v / layer.gene_exp.sum(axis=1)
    v = np.log(v)
    scat = plt.scatter(layer.obsm['spatial'][:, 0], layer.obsm['spatial'][:, 1], linewidth=0, s=150, marker=".", c=v,
                       cmap=cmap, vmin=vmin, vmax=vmax)
    plt.gca().set_title(title, fontsize=26)
    cbar = plt.colorbar(scat, shrink=0.8)  # , location='right')
    cbar.ax.set_ylabel('log counts', fontsize=22)
    plt.axis('off')
    plt.gca().invert_yaxis()

    if draw_contours:
        for l in ['Layer{}'.format(i) for i in [1, 3, 5]] + ['WM']:
            contours = get_scatter_contours(layer, adatas[sample_list[2 * 4 + 1]].obs['layer_guess_reordered'], [l])
            for k in range(len(contours)):
                plt.plot(layer.obsm['spatial'][contours[k], 0], layer.obsm['spatial'][contours[k], 1], 'lime', lw=4,
                         alpha=0.6)

    plt.gca().text(110, 150, 'L1')
    plt.gca().text(110, 220, 'L2')
    plt.gca().text(110, 260, 'L3')
    plt.gca().text(110, 305, 'L4')
    plt.gca().text(110, 340, 'L5')
    plt.gca().text(110, 380, 'L6')
    plt.gca().text(110, 425, 'WM')

    plt.show()
    return


def cluster_adata(adata, n_clusters=7, sample_name='', use_nmf=False):
    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy, inplace=True)
    sc.pp.log1p(adata_copy)
    sc.pp.highly_variable_genes(adata_copy, flavor="seurat", n_top_genes=2000)
    sc.pp.pca(adata_copy)

    if use_nmf:
        model = sklearn.decomposition.NMF(n_components=50)
        adata_copy.obsm['X_pca'] = model.fit_transform(adata_copy.X)

    cluster_labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=500).fit_predict(adata_copy.obsm['X_pca'])

    adata_copy.obs['my_clusters'] = pd.Series(
        1 + match_cluster_labels(adata_copy.obs['layer_guess_reordered'], cluster_labels), index=adata_copy.obs.index,
        dtype='category')

    ari = sklearn.metrics.adjusted_rand_score(adata_copy.obs['layer_guess_reordered'], adata_copy.obs['my_clusters'])
    print('ARI', ari)
    adata.obs['my_clusters'] = adata_copy.obs['my_clusters'].copy()
    return


def draw_spatial(adata, clusters='my_clusters', sample_name='', draw_contours=False):
    fig = plt.figure(figsize=(12, 10))
    ax = sc.pl.spatial(adata, color=clusters, spot_size=5, show=False,
                       palette=['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d'],
                       ax=plt.gca())
    ax[0].axis('off')
    ari = sklearn.metrics.adjusted_rand_score(adata.obs['layer_guess_reordered'], adata.obs[clusters])
    ax[0].legend(title='Cluster', bbox_to_anchor=(0.95, 0.85), fontsize=20, title_fontsize=20)
    ax[0].set_title('{}: ARI={:.2f}'.format(sample_name, ari), fontsize=26)
    if draw_contours:
        for l in ['Layer{}'.format(i) for i in [1, 3, 5]] + ['WM']:
            contours = get_scatter_contours(adata, adata.obs['layer_guess_reordered'], [l])
            for k in range(len(contours)):
                plt.plot(adata.obsm['spatial'][contours[k], 0], adata.obsm['spatial'][contours[k], 1], 'lime',
                         # dict(zip(['Layer{}'.format(i) for i in range(1,7)]+['WM'],adata.uns['layer_guess_reordered_colors']))[l],
                         lw=4, alpha=0.6)
    plt.gca().text(100, 150, 'L1')
    plt.gca().text(100, 220, 'L2')
    plt.gca().text(100, 260, 'L3')
    plt.gca().text(100, 305, 'L4')
    plt.gca().text(100, 340, 'L5')
    plt.gca().text(100, 380, 'L6')
    plt.gca().text(100, 425, 'WM')
    plt.show()


def callback(
    slices,
    new_slices,
    data_expression_ax,
    latent_expression_ax,
    n_views=2,
    gene_idx=1897,
    s=200,
    include_legend=False,
):
    markers = [".", "+", "^"]
    colors = ["blue", "orange"]

    data_expression_ax.cla()
    latent_expression_ax.cla()
    data_expression_ax.set_title("Observed data")
    latent_expression_ax.set_title("Aligned data")

    latent_Xs = []
    Xs = []
    Ys = []
    markers_list = []
    viewname_list = []

    data = slices[0].concatenate(slices[1])
    n_samples_list = [slices[0].shape[0], slices[1].shape[0]]
    view_idx = [
        np.arange(slices[0].shape[0]),
        np.arange(slices[0].shape[0], slices[0].shape[0] + slices[1].shape[0]),
    ]
    X1 = data[data.obs.batch == "0"].obsm["spatial"]
    X2 = data[data.obs.batch == "1"].obsm["spatial"]
    Y1 = np.array(data[data.obs.batch == "0"].X.todense())
    Y2 = np.array(data[data.obs.batch == "1"].X.todense())
    Y1 = (Y1 - Y1.mean(0)) / Y1.std(0)
    Y2 = (Y2 - Y2.mean(0)) / Y2.std(0)
    X = np.concatenate([X1, X2])
    Y = np.concatenate([Y1, Y2])

    data2 = new_slices[0].concatenate(new_slices[1])
    X11 = data2[data2.obs.batch == "0"].obsm["spatial"]
    X22 = data2[data2.obs.batch == "1"].obsm["spatial"]
    Y11 = np.array(data2[data2.obs.batch == "0"].X.todense())
    Y22 = np.array(data2[data2.obs.batch == "1"].X.todense())
    Y11 = (Y11 - Y11.mean(0)) / Y11.std(0)
    Y22 = (Y22 - Y22.mean(0)) / Y22.std(0)
    X_new = np.concatenate([X11, X22])
    Y_new = np.concatenate([Y11, Y22])

    for vv in range(n_views):
        ## Data
        Xs.append(X[view_idx[vv]])
        ## Latents
        latent_Xs.append(X_new[view_idx[vv]])
        Ys.append(Y[view_idx[vv], gene_idx])
        markers_list.append([markers[vv]] * slices[vv].obsm['spatial'].shape[0])
        viewname_list.append(
            ["Observation {}".format(vv + 1)] * slices[vv].obsm['spatial'].shape[0]
        )

    Xs = np.concatenate(Xs, axis=0)
    latent_Xs = np.concatenate(latent_Xs, axis=0)
    Ys = np.concatenate(Ys)
    markers_list = np.concatenate(markers_list)
    viewname_list = np.concatenate(viewname_list)

    data_df = pd.DataFrame(
        {
            "X1": Xs[:, 0],
            "X2": Xs[:, 1],
            "Y": Ys,
            "marker": markers_list,
            "view": viewname_list,
        }
    )

    latent_df = pd.DataFrame(
        {
            "X1": latent_Xs[:, 0],
            "X2": latent_Xs[:, 1],
            "Y": Ys,
            "marker": markers_list,
            "view": viewname_list,
        }
    )

    plt.sca(data_expression_ax)
    g = sns.scatterplot(
        data=data_df,
        x="X1",
        y="X2",
        hue="Y",
        style="view",
        ax=data_expression_ax,
        s=s,
        linewidth=0.5,
        edgecolor="black",
        palette="viridis",
    )
    if not include_legend:
        g.legend_.remove()

    plt.sca(latent_expression_ax)
    g = sns.scatterplot(
        data=latent_df,
        x="X1",
        y="X2",
        hue="Y",
        style="view",
        ax=latent_expression_ax,
        s=s,
        linewidth=0.5,
        edgecolor="black",
        palette="viridis",
    )
    if not include_legend:
        g.legend_.remove()


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