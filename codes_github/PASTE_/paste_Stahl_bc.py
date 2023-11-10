import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scanpy as sc
import paste as pst
import ot
import time
import plotly.express as px
import pandas as pd

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

# plot raw slices
fig, axs = plt.subplots(2, 2,figsize=(7,7))
pst.plot_slice(slice1,slice_colors[0],ax=axs[0,0])
pst.plot_slice(slice2,slice_colors[1],ax=axs[0,1])
pst.plot_slice(slice3,slice_colors[2],ax=axs[1,0])
pst.plot_slice(slice4,slice_colors[3],ax=axs[1,1])

# save_path = "../../results/filtered_stahl_bc.png"
# plt.savefig(save_path)
plt.show()

# paste_alignment
start = time.time()
pi12 = pst.pairwise_align(slice1, slice2, backend=ot.backend.TorchBackend(), use_gpu=True)
pi23 = pst.pairwise_align(slice2, slice3, backend=ot.backend.TorchBackend(), use_gpu=True)
pi34 = pst.pairwise_align(slice3, slice4, backend=ot.backend.TorchBackend(), use_gpu=True)
print('Alignment Runtime: ' + str(time.time() - start))

# To visualize the alignment you can stack the slices
# according to the alignment pi
slices, pis = [slice1, slice2, slice3, slice4], [pi12, pi23, pi34]
new_slices = pst.stack_slices_pairwise(slices, pis)

# plot paste_alignment results
plt.figure(figsize=(7,7))
for i in range(len(new_slices)):
    pst.plot_slice(new_slices[i],slice_colors[i], s=400)
plt.legend(handles=[mpatches.Patch(color=slice_colors[0], label='1'),
                    mpatches.Patch(color=slice_colors[1], label='2'),
                    mpatches.Patch(color=slice_colors[2], label='3'),
                    mpatches.Patch(color=slice_colors[3], label='4')])
plt.gca().invert_yaxis()
plt.axis('off')
# save_path = "../../results/stahl_bc/paste_alignment_stahl_bc.png"
# plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(2, 2,figsize=(7,7))
pst.plot_slice(new_slices[0], slice_colors[0], ax=axs[0,0])
pst.plot_slice(new_slices[1], slice_colors[1], ax=axs[0,0])
pst.plot_slice(new_slices[1], slice_colors[1], ax=axs[0,1])
pst.plot_slice(new_slices[2], slice_colors[2], ax=axs[0,1])
pst.plot_slice(new_slices[2], slice_colors[2], ax=axs[1,0])
pst.plot_slice(new_slices[3], slice_colors[3], ax=axs[1,0])
fig.delaxes(axs[1,1])
# save_path = "../../results/stahl_bc/paste_alignment_individual_stahl_bc.png"
# plt.savefig(save_path)
plt.show()


# paste_integration
# We have to reload the slices as pairwise_alignment modifies the slices.
slices = load_slices(data_dir, slice_names=slice_names)
slice1, slice2, slice3, slice4 = slices
print([x.obsm['spatial'].shape for x in slices])

# Construct a center slice
# choose one of the slices as the coordinate reference for the center slice,
# i.e. the center slice will have the same number of spots as this slice and
# the same coordinates.
initial_slice = slice1.copy()
slices = [slice1, slice2, slice3, slice4]
lmbda = len(slices)*[1/len(slices)]  # set hyperparameter to be uniform

# Possible to pass in an initial pi (as keyword argument pis_init)
# to improve performance, see Tutorial.ipynb notebook for more details.
start = time.time()
center_slice, pis = pst.center_align(initial_slice, slices, lmbda, random_seed=5,
                                     backend=ot.backend.TorchBackend(), use_gpu=True)
print('Integration Runtime: ' + str(time.time() - start))

# The low dimensional representation of our center slice is held
# in the matrices W and H, which can be used for downstream analyses
W = center_slice.uns['paste_W']
H = center_slice.uns['paste_H']
# print(W.shape, H.shape)

all_slices = pst.stack_slices_center(center_slice, slices, pis)
center, new_slices = all_slices[0], all_slices[1]
center_color = 'orange'
slices_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']

# plot paste_integration results
plt.figure(figsize=(7,7))
pst.plot_slice(center,center_color,s=400)
for i in range(len(new_slices)):
    pst.plot_slice(new_slices[i],slices_colors[i],s=400)

plt.legend(handles=[mpatches.Patch(color=slices_colors[0], label='1'),
                    mpatches.Patch(color=slices_colors[1], label='2'),
                    mpatches.Patch(color=slices_colors[2], label='3'),
                    mpatches.Patch(color=slices_colors[3], label='4')])
plt.gca().invert_yaxis()
plt.axis('off')
# save_path = "../../results/stahl_bc/paste_integration_stahl_bc.png"
# plt.savefig(save_path)
plt.show()

fig, axs = plt.subplots(2, 2,figsize=(7,7))
pst.plot_slice(center,center_color,ax=axs[0,0])
pst.plot_slice(new_slices[0],slices_colors[0],ax=axs[0,0])

pst.plot_slice(center,center_color,ax=axs[0,1])
pst.plot_slice(new_slices[1],slices_colors[1],ax=axs[0,1])

pst.plot_slice(center,center_color,ax=axs[1,0])
pst.plot_slice(new_slices[2],slices_colors[2],ax=axs[1,0])

pst.plot_slice(center,center_color,ax=axs[1,1])
pst.plot_slice(new_slices[3],slices_colors[3],ax=axs[1,1])
# save_path = "../../results/stahl_bc/paste_integration_individual_stahl_bc.png"
# plt.savefig(save_path)
plt.show()