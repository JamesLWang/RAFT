import matplotlib.pyplot as plt

mask_percentage = [1, 5, 10, 15, 20]
auroc_opt = []
auroc_roberta = []
perplexity_opt = []
perplexity_roberta = []

fig, ax1 = plt.subplots()

ax1.plot(mask_percentage, auroc_opt, 'o-', color='brown', label='OPT-2.7B')
ax1.plot(mask_percentage, auroc_roberta, 'o-', color='orange', label='RoBERTa-Large')
ax1.set_xlabel('Mask Percentage')
ax1.set_ylabel('AUROC')
ax1.legend(loc='center right')

ax2 = ax1.twinx()
ax2.plot(mask_percentage, perplexity_opt, 'o--', color='blue', label='OPT-2.7B Perplexity')
ax2.plot(mask_percentage, perplexity_roberta, 'o--', color='green', label='RoBERTa-large Perplexity')
ax2.set_ylabel('Perplexity')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='best')

plt.show()
