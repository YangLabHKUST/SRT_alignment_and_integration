import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

data_dir = '../data/3dst/data/preprocessed/'
save_dir = '../results/RA/'

n_samples = 6
n_slices = [4, 7, 4, 5, 3, 4]

cls = ['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Collagenous/Adipose', 'Infiltrate;Unknown', 'Unknown']
colors = ['deeppink', 'royalblue', 'blueviolet', 'forestgreen', 'darkgray', 'gray']
sp_cmap = {cls: color for cls, color in zip(cls, colors)}

cls_list = [['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Collagenous/Adipose', 'Unknown'],
            ['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Unknown'],
            ['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Infiltrate;Unknown'],
            ['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Collagenous/Adipose', 'Infiltrate;Unknown'],
            ['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Collagenous/Adipose', 'Infiltrate;Unknown'],
            ['Infiltrate', 'Synovial', 'Infiltrate;Synovial', 'Collagenous/Adipose']]

fig_size = [10, 15, 10, 12, 8.5, 10]
spot_size = [100, 100, 40, 30, 50, 80]

for draw in range(n_samples):

    slices = [ad.read_h5ad(data_dir + f"RA{draw+1}/RA{draw+1}_slice{i+1}.h5ad") for i in range(n_slices[draw])]

    fig, axs = plt.subplots(1, n_slices[draw], figsize=(fig_size[draw], 2))
    for i in range(len(slices)):
        real_colors = list(slices[i].obs['annotations'].astype('str').map(sp_cmap))
        axs[i].scatter(slices[i].obsm['spatial'][:, 0], slices[i].obsm['spatial'][:, 1], linewidth=0,
                       s=spot_size[draw], marker=".", color=real_colors)
        axs[i].set_title(f'RA{draw + 1} Slice{i + 1}', size=12)
        axs[i].invert_yaxis()
        axs[i].axis('off')

    axs[n_slices[draw] - 1].legend(handles=[mpatches.Patch(color=sp_cmap[spot_class], label=spot_class) for spot_class in cls_list[draw]],
                                   fontsize=8, title='Clusters', title_fontsize=10, bbox_to_anchor=(1, 1))
    axs[n_slices[draw] - 1].axis('off')

    plt.gcf().subplots_adjust(left=0.05, top=None, bottom=None, right=0.8)
    # plt.savefig(save_dir + f'RA{draw + 1}_annotation.png', dpi=300)
plt.show()
