import numpy as np
import matplotlib.pyplot as plt


slice_map = {0:'A',1:'B',2:'C',3:'D'}
sample_map = {0:'A',1:'B',2:'C'}

for i in range(3):
    results_dict = np.load('../../results/accuracy/ratio_accuracy_Sample{}_DLPFC.npz'.format(sample_map[i]))


    all_ratio = results_dict["all_ratio"]
    plt.figure()
    labels = ['AB', 'BC', 'CD']
    raw, paste, paste2, gpsa, staligner = all_ratio
    bar_width = 0.15
    x = range(len(labels))
    # plt.bar(x, raw, width=bar_width, label='raw')
    # plt.bar([i + bar_width for i in x], paste, width=bar_width, label='PASTE')
    # plt.bar([i + 2 * bar_width for i in x], paste2, width=bar_width, label='PASTE2')
    # plt.bar([i + 3 * bar_width for i in x], gpsa, width=bar_width, label='GPSA')
    # plt.bar([i + 4 * bar_width for i in x], staligner, width=bar_width, label='STAligner')
    # plt.xticks([i + 2 * bar_width for i in x], labels, fontsize=16)
    plt.bar(x, paste, width=bar_width, label='PASTE')
    plt.bar([i + bar_width for i in x], paste2, width=bar_width, label='PASTE2')
    plt.bar([i + 2 * bar_width for i in x], gpsa, width=bar_width, label='GPSA')
    plt.bar([i + 3 * bar_width for i in x], staligner, width=bar_width, label='STAligner')
    plt.xticks([i + 1.5 * bar_width for i in x], labels, fontsize=16)
    plt.title('Sample {}'.format(sample_map[i]), fontsize=18)
    plt.legend(loc=(0.77,0))
    plt.ylabel('ratio', fontsize=16)
    plt.savefig('../../results/accuracy/ratio_Sample{}_DLPFC.png'.format(sample_map[i]))
    plt.show()


    all_accu = results_dict["all_accu"]
    plt.figure()
    labels = ['AB', 'BC', 'CD']
    raw, paste, paste2, gpsa, staligner = all_accu
    bar_width = 0.15
    x = range(len(labels))
    # plt.bar(x, raw, width=bar_width, label='raw')
    # plt.bar([i + bar_width for i in x], paste, width=bar_width, label='PASTE')
    # plt.bar([i + 2 * bar_width for i in x], paste2, width=bar_width, label='PASTE2')
    # plt.bar([i + 3 * bar_width for i in x], gpsa, width=bar_width, label='GPSA')
    # plt.bar([i + 4 * bar_width for i in x], staligner, width=bar_width, label='STAligner')
    # plt.xticks([i + 2 * bar_width for i in x], labels, fontsize=16)
    plt.bar(x, paste, width=bar_width, label='PASTE')
    plt.bar([i + bar_width for i in x], paste2, width=bar_width, label='PASTE2')
    plt.bar([i + 2 * bar_width for i in x], gpsa, width=bar_width, label='GPSA')
    plt.bar([i + 3 * bar_width for i in x], staligner, width=bar_width, label='STAligner')
    plt.xticks([i + 1.5 * bar_width for i in x], labels, fontsize=16)
    plt.title('Sample {}'.format(sample_map[i]), fontsize=18)
    plt.legend(loc=(0.77, 0))
    plt.ylabel('accuracy', fontsize=16)
    plt.savefig('../../results/accuracy/accuracy_Sample{}_DLPFC.png'.format(sample_map[i]))
    plt.show()


    all_mean_accu = results_dict["all_mean_accu"]
    plt.figure()
    # methods = ['raw', 'PASTE', 'PASTE2', 'GPSA', 'STAligner']
    # plt.bar(methods, all_mean_accu, width=0.5)
    methods = ['PASTE', 'PASTE2', 'GPSA', 'STAligner']
    plt.bar(methods, all_mean_accu[1:], width=0.4)
    plt.title('Sample {}'.format(sample_map[i]), fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel('mean accuracy', fontsize=16)
    plt.savefig('../../results/accuracy/mean_accuracy_Sample{}_DLPFC.png'.format(sample_map[i]))
    plt.show()


for i in range(3):
    results_dict = np.load('../../results/accuracy/ratio_accuracy_Sample{}_partial_DLPFC_0.85.npz'.format(sample_map[i]))


    all_ratio = results_dict["all_ratio"]
    plt.figure()
    labels = ['AB', 'BC', 'CD']
    raw, paste, paste2, gpsa, staligner = all_ratio
    bar_width = 0.15
    x = range(len(labels))
    # plt.bar(x, raw, width=bar_width, label='raw')
    # plt.bar([i + bar_width for i in x], paste, width=bar_width, label='PASTE')
    # plt.bar([i + 2 * bar_width for i in x], paste2, width=bar_width, label='PASTE2')
    # plt.bar([i + 3 * bar_width for i in x], gpsa, width=bar_width, label='GPSA')
    # plt.bar([i + 4 * bar_width for i in x], staligner, width=bar_width, label='STAligner')
    # plt.xticks([i + 2 * bar_width for i in x], labels, fontsize=16)
    plt.bar(x, paste, width=bar_width, label='PASTE')
    plt.bar([i + bar_width for i in x], paste2, width=bar_width, label='PASTE2')
    plt.bar([i + 2 * bar_width for i in x], gpsa, width=bar_width, label='GPSA')
    plt.bar([i + 3 * bar_width for i in x], staligner, width=bar_width, label='STAligner')
    plt.xticks([i + 1.5 * bar_width for i in x], labels, fontsize=16)
    plt.title('Sample {}'.format(sample_map[i]), fontsize=18)
    plt.legend(loc=(0.77,0))
    plt.ylabel('ratio', fontsize=16)
    plt.savefig('../../results/accuracy/0.85_ratio_Sample{}_partial_DLPFC.png'.format(sample_map[i]))
    plt.show()


    all_accu = results_dict["all_accu"]
    plt.figure()
    labels = ['AB', 'BC', 'CD']
    raw, paste, paste2, gpsa, staligner = all_accu
    bar_width = 0.15
    x = range(len(labels))
    # plt.bar(x, raw, width=bar_width, label='raw')
    # plt.bar([i + bar_width for i in x], paste, width=bar_width, label='PASTE')
    # plt.bar([i + 2 * bar_width for i in x], paste2, width=bar_width, label='PASTE2')
    # plt.bar([i + 3 * bar_width for i in x], gpsa, width=bar_width, label='GPSA')
    # plt.bar([i + 4 * bar_width for i in x], staligner, width=bar_width, label='STAligner')
    # plt.xticks([i + 2 * bar_width for i in x], labels, fontsize=16)
    plt.bar(x, paste, width=bar_width, label='PASTE')
    plt.bar([i + bar_width for i in x], paste2, width=bar_width, label='PASTE2')
    plt.bar([i + 2 * bar_width for i in x], gpsa, width=bar_width, label='GPSA')
    plt.bar([i + 3 * bar_width for i in x], staligner, width=bar_width, label='STAligner')
    plt.xticks([i + 1.5 * bar_width for i in x], labels, fontsize=16)
    plt.title('Sample {}'.format(sample_map[i]), fontsize=18)
    plt.legend(loc=(0.77, 0))
    plt.ylabel('accuracy', fontsize=16)
    plt.savefig('../../results/accuracy/0.85_accuracy_Sample{}_partial_DLPFC.png'.format(sample_map[i]))
    plt.show()


    all_mean_accu = results_dict["all_mean_accu"]
    plt.figure()
    # methods = ['raw', 'PASTE', 'PASTE2', 'GPSA', 'STAligner']
    # plt.bar(methods, all_mean_accu, width=0.5)
    methods = ['PASTE', 'PASTE2', 'GPSA', 'STAligner']
    plt.bar(methods, all_mean_accu[1:], width=0.4)
    plt.title('Sample {}'.format(sample_map[i]), fontsize=18)
    plt.xticks(fontsize=16)
    plt.ylabel('mean accuracy', fontsize=16)
    plt.savefig('../../results/accuracy/0.85_mean_accuracy_Sample{}_partial_DLPFC.png'.format(sample_map[i]))
    plt.show()


