import time
import paste as pst
import matplotlib
from paste2.PASTE2 import *
from paste2.model_selection import *
from paste2.projection import *
from paste2.helper import *
from codes_github.utils import *


fn1 = '../../data/mouse_brain/sample1'
fn2 = '../../data/mouse_brain/sample2'
slice1 = sc.read_visium(fn1)
slice1.var_names_make_unique()
slice1 = process_data(slice1, n_top_genes=6000)
slice2 = sc.read_visium(fn2)
slice1.var_names_make_unique()
slice2 = process_data(slice2, n_top_genes=6000)

# Pairwise align the slices
start = time.time()
s = select_overlap_fraction(slice1, slice2)
ss = [s]
pi0 = match_spots_using_spatial_heuristic(slice1.obsm['spatial'],slice2.obsm['spatial'], use_ot=True)
pi12 = partial_pairwise_align(slice1, slice2, min(s, 0.99), G_init=pi0)
print('Alignment Runtime: ' + str(time.time() - start))

# To visualize the alignment you can stack the slices
# according to the alignment pi
slices, pis = [slice1, slice2], [pi12]
new_slices = pst.stack_slices_pairwise(slices, pis)

pis_dict = {}
for i, matrix in enumerate(pis):
    pis_dict[f'pi_{i}'] = matrix
pis_dict['s'] = ss
np.savez('../../results/mouse_brain/paste2_pis_mouse_brain.npz')

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

fig = plt.figure(figsize=(14, 7))
data_expression_ax = fig.add_subplot(121, frameon=False)
latent_expression_ax = fig.add_subplot(122, frameon=False)
idx = (slice1.concatenate(slice2)).var_names.tolist().index('Pcp2')
callback([slice1, slice2], new_slices, s=40, data_expression_ax=data_expression_ax,
         latent_expression_ax=latent_expression_ax, gene_idx=idx)
latent_expression_ax.set_title("Aligned data, PASTE2")
latent_expression_ax.set_axis_off()
data_expression_ax.set_axis_off()

plt.tight_layout()
plt.savefig("../../results/mouse_brain/paste2_mouse_brain.png")
plt.show()