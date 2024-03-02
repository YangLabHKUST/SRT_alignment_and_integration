import pickle
import numpy as np
import matplotlib.pyplot as plt


save_dir = '../results/RA/'
save = False

aris, b_asws, b_pcrs, kbets, g_conns = [], [], [], [], []

result_dict_path_list = ['STAligner/STAligner_sample_specific_results_dict.pkl',
                         'STAligner/STAligner_joint_samples_results_dict.pkl',
                         'STitch3D/PASTE_alignment+STitch3D_results_dict.pkl',
                         'STitch3D/PASTE2+STitch3D_results_dict.pkl',
                         'PASTE_integration/PASTE_integration_results_dict.pkl']

for i in range(len(result_dict_path_list)):
    with open(save_dir + result_dict_path_list[i], 'rb') as file:
        results_dict = pickle.load(file)
    aris.append(results_dict['ARIs'])
    if i != 4:
        b_asws.append(results_dict['Batch_ASWs'])
        b_pcrs.append(results_dict['Batch_PCRs'])
        kbets.append(results_dict['kBETs'])
        g_conns.append(results_dict['Graph_connectivities'])

scores = [aris, b_asws, b_pcrs, kbets, g_conns]
metric_list = ['ARI', 'Batch ASW', 'Batch PCR', 'kBET', 'Graph connectivity']

for j in range(len(metric_list)):

    fig, axs = plt.subplots(figsize=(11, 3))

    if j == 0:
        methods = ['STAligner (sample specific)', 'STAligner (joint)', 'PASTE_alignment+STitch3D',
                   'PASTE2+STitch3D', 'PASTE_integration']
        labels = ['STAligner\n(sample specific)', 'STAligner\n(joint)', 'PASTE_alignment\n+STitch3D',
                  'PASTE2\n+STitch3D', 'PASTE_integration']
        color_list = ['deepskyblue', 'dodgerblue', 'fuchsia', 'm', 'crimson']
        sample_list = ['RA1', 'RA2', 'RA3', 'RA4', 'RA5', 'RA6']
    else:
        methods = ['STAligner (sample specific)', 'STAligner (joint)', 'PASTE_alignment+STitch3D',
                   'PASTE2+STitch3D']
        labels = ['STAligner\n(sample specific)', 'STAligner\n(joint)', 'PASTE_alignment\n+STitch3D',
                  'PASTE2\n+STitch3D']
        color_list = ['deepskyblue', 'dodgerblue', 'fuchsia', 'm']
        sample_list = ['RA1', 'RA2', 'RA3', 'RA4', 'RA5', 'RA6']

    for i, method in enumerate(methods):

        positions = np.arange(6) * (len(methods) + 1) + i
        axs.bar(positions, scores[j][i], width=0.8, color=color_list[i], capsize=5)

    axs.set_title(f'{metric_list[j]}', fontsize=16)

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_linewidth(1)
    axs.spines['bottom'].set_linewidth(1)

    axs.set_xticks([(len(methods)/2-0.5) + (len(methods)+1) * i for i in range(6)])
    axs.set_xticklabels(sample_list, fontsize=12)
    axs.tick_params(axis='y', labelsize=12)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_list[i], label=label)
               for i, label in enumerate(labels)]
    axs.legend(handles=handles, loc=(1, 0.3), fontsize=10)
    plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.8)

    if save:
        save_path = save_dir + f"{metric_list[j]}.png"
        plt.savefig(save_path, dpi=500)
plt.show()


# RA2 ARI
fig, axs = plt.subplots(figsize=(6, 4.5))
RA2_aris = []
for i in range(len(aris)):
    RA2_aris.append(aris[i][1])

methods = ['STAligner (sample specific)', 'STAligner (joint)', 'PASTE_alignment+STitch3D',
           'PASTE2+STitch3D', 'PASTE_integration']
labels = ['STAligner\n(sample specific)', 'STAligner\n(joint)', 'PASTE_alignment\n+STitch3D',
          'PASTE2\n+STitch3D', 'PASTE_integration']
color_list = ['deepskyblue', 'dodgerblue', 'fuchsia', 'm', 'crimson']

plt.bar(labels, RA2_aris, color=color_list, width=0.5)

plt.title('ARI', fontsize=18)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

plt.xticks(range(len(labels)), labels, fontsize=9.5)
axs.tick_params(axis='y', labelsize=12)

if save:
    save_path = save_dir + f"RA2_ARI.png"
    plt.savefig(save_path)
plt.show()


# batch correction
b_asws = np.array(b_asws)
b_pcrs = np.array(b_pcrs)
kbets = np.array(kbets)
g_conns = np.array(g_conns)
mean = (b_asws + b_pcrs + kbets + g_conns) / 4

methods = ['STAligner (sample specific)', 'STAligner (joint)', 'PASTE_alignment+STitch3D',
           'PASTE2+STitch3D']
labels = ['STAligner\n(sample specific)', 'STAligner\n(joint)', 'PASTE_alignment\n+STitch3D',
          'PASTE2\n+STitch3D']
color_list = ['deepskyblue', 'dodgerblue', 'fuchsia', 'm']
sample_list = ['RA1', 'RA2', 'RA3', 'RA4', 'RA5', 'RA6']

fig, axs = plt.subplots(figsize=(11, 3))

for i, method in enumerate(methods):
    positions = np.arange(6) * (len(methods) + 1) + i
    axs.bar(positions, mean[i], width=0.8, color=color_list[i], capsize=5)

axs.set_title(f'Batch Correction Performance', fontsize=16)

axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_linewidth(1)
axs.spines['bottom'].set_linewidth(1)

axs.set_xticks([(len(methods) / 2 - 0.5) + (len(methods) + 1) * i for i in range(6)])
axs.set_xticklabels(sample_list, fontsize=12)
axs.tick_params(axis='y', labelsize=12)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_list[i], label=label)
           for i, label in enumerate(labels)]
axs.legend(handles=handles, loc=(1, 0.3), fontsize=10)
plt.gcf().subplots_adjust(left=0.1, top=0.9, bottom=0.15, right=0.8)

if save:
    save_path = save_dir + f"batch_correction.png"
    plt.savefig(save_path, dpi=500)
plt.show()
