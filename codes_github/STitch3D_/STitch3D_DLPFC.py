import scipy.io
import seaborn as sns
from paste2.model_selection import *
from paste2.helper import *
import STitch3D

import warnings
warnings.filterwarnings("ignore")

slice_map = {0:'A',1:'B',2:'C',3:'D'}
sample_map = {0:'A',1:'B',2:'C'}

np.random.seed(1234)

mat = scipy.io.mmread("../../data/10x_DLPFC/GSE144136_GeneBarcodeMatrix_Annotated.mtx")

meta = pd.read_csv("../../data/10x_DLPFC/GSE144136_CellNames.csv", index_col=0)
meta.index = meta.x.values
group = [i.split('.')[1].split('_')[0] for i in list(meta.x.values)]
condition = [i.split('.')[1].split('_')[1] for i in list(meta.x.values)]
celltype = [i.split('.')[0] for i in list(meta.x.values)]
meta["group"] = group
meta["condition"] = condition
meta["celltype"] = celltype

genename = pd.read_csv("../../data/10x_DLPFC/GSE144136_GeneNames.csv", index_col=0)
genename.index = genename.x.values

adata_ref = ad.AnnData(X=mat.tocsr().T)
adata_ref.obs = meta
adata_ref.var = genename
adata_ref = adata_ref[adata_ref.obs.condition.values.astype(str)=="Control", :]

sample_list = ["151507", "151508", "151509","151510", "151669", "151670","151671", "151672", "151673","151674", "151675", "151676"]
# adatas = {sample:sc.read_h5ad('../../data/DLPFC/{0}_preprocessed.h5'.format(sample)) for sample in sample_list}
sample_groups = [["151507", "151508", "151509","151510"], [ "151669", "151670","151671", "151672"], [ "151673","151674", "151675", "151676"]]
# layer_groups = [[adatas[sample_groups[j][i]] for i in range(len(sample_groups[j]))] for j in range(len(sample_groups))]
layer_to_color_map = {'Layer{0}'.format(i+1):sns.color_palette()[i] for i in range(6)}
layer_to_color_map['WM'] = sns.color_palette()[6]

# present the experiment of STitch3D
# using DLPFC dataset which has been aligned by paste_alignment as an example
for i in range(len(sample_groups)):
    adata_st_list = []
    adata_st_list_raw = []
    for j in range(len(sample_groups[i])):
        slice_raw = sc.read_h5ad('../../data/DLPFC/{0}_preprocessed.h5'.format(sample_groups[i][j]))
        # if j % 2 == 0:
        #     slice_raw = partial_cut(slice_raw, 0.85, is_left=True)
        # else:
        #     slice_raw = partial_cut(slice_raw, 0.85, is_left=False)
        adata_st_list_raw.append(slice_raw)

        # load slices aligned by paste_alignment
        slice = sc.read_h5ad("../../results/stitch3d_use/DLPFC/paste_alignment_Sample{}_slice{}_DLPFC.h5ad".format(sample_map[i], slice_map[j]))
        slice.obsm['spatial_aligned'] = slice.obsm['spatial']
        del slice.obsm['spatial']
        adata_st_list.append(slice)

    celltype_list_use = ['Astros_1', 'Astros_2', 'Astros_3', 'Endo', 'Micro/Macro',
                         'Oligos_1', 'Oligos_2', 'Oligos_3',
                         'Ex_1_L5_6', 'Ex_2_L5', 'Ex_3_L4_5', 'Ex_4_L_6', 'Ex_5_L5',
                         'Ex_6_L4_6', 'Ex_7_L4_6', 'Ex_8_L5_6', 'Ex_9_L5_6', 'Ex_10_L2_4']

    adata_st, adata_basis = STitch3D.utils.preprocess(adata_st_list,
                                                      adata_ref,
                                                      celltype_ref=celltype_list_use,
                                                      sample_col="group",
                                                      slice_dist_micron=[10., 300., 10.],
                                                      n_hvg_group=500)

    model = STitch3D.model.Model(adata_st, adata_basis)

    model.train()

    save_path = "../../results/stitch3d_use/downstream/DLPFC/Sample{}".format(sample_map[i])
    result = model.eval(adata_st_list_raw, save=True, output_path=save_path)


