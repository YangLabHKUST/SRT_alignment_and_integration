import matplotlib.patches as mpatches
import time
import paste as pst
from ..src.paste2.PASTE2 import *
from ..src.paste2.model_selection import *
from ..src.paste2.projection import *
from ..src.paste2.helper import *
from ..utils import *


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
        slices.append(slice_i)
    return slices


slice_names = ['stahl_bc_slice1', 'stahl_bc_slice2', 'stahl_bc_slice3', 'stahl_bc_slice4']
slices = load_slices(data_dir, slice_names=slice_names)
slice1, slice2, slice3, slice4 = slices

# filtered image
slice_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']

# Pairwise align the slices
start = time.time()
s1 = select_overlap_fraction(slice1, slice2)
s2 = select_overlap_fraction(slice2, slice3)
s3 = select_overlap_fraction(slice3, slice4)
ss = [s1, s2, s3]
print(ss)
# pi0 = match_spots_using_spatial_heuristic(slice1.obsm['spatial'], slice2.obsm['spatial'])
pi12 = partial_pairwise_align(slice1, slice2, s1)#, G_init=pi0)
# pi0 = match_spots_using_spatial_heuristic(slice2.obsm['spatial'], slice3.obsm['spatial'])
pi23 = partial_pairwise_align(slice2, slice3, s2)#, G_init=pi0)
# pi0 = match_spots_using_spatial_heuristic(slice3.obsm['spatial'], slice4.obsm['spatial'])
pi34 = partial_pairwise_align(slice3, slice4, s3)#, G_init=pi0)
print('Alignment Runtime: ' + str(time.time() - start))


# To visualize the alignment you can stack the slices
# according to the alignment pi
slices, pis = [slice1, slice2, slice3, slice4], [pi12, pi23, pi34]
new_slices = partial_stack_slices_pairwise(slices, pis)

plt.figure(figsize=(7,7))
for i in range(len(new_slices)):
    pst.plot_slice(new_slices[i],slice_colors[i], s=400)
plt.legend(handles=[mpatches.Patch(color=slice_colors[0], label='1'),
                    mpatches.Patch(color=slice_colors[1], label='2'),
                    mpatches.Patch(color=slice_colors[2], label='3'),
                    mpatches.Patch(color=slice_colors[3], label='4')])
plt.gca().invert_yaxis()
plt.axis('off')
# save_path = "../../results/stahl_bc/paste2_stahl_bc.png"
# plt.savefig(save_path)
# plt.show()

fig, axs = plt.subplots(2, 2,figsize=(7,7))
pst.plot_slice(new_slices[0], slice_colors[0], ax=axs[0,0])
pst.plot_slice(new_slices[1], slice_colors[1], ax=axs[0,0])
pst.plot_slice(new_slices[1], slice_colors[1], ax=axs[0,1])
pst.plot_slice(new_slices[2], slice_colors[2], ax=axs[0,1])
pst.plot_slice(new_slices[2], slice_colors[2], ax=axs[1,0])
pst.plot_slice(new_slices[3], slice_colors[3], ax=axs[1,0])
fig.delaxes(axs[1,1])
# save_path = "../../results/stahl_bc/paste2_individual_stahl_bc.png"
# plt.savefig(save_path)
plt.show()