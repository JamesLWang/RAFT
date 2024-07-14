import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# XSum
data_gpt35 = np.array([
    [0.2793,0.2023,0.2187],
    [0.2408,0.1202,0.1222],
    [0.0010,0.0017,0.0023]
])

data_llama = np.array([
    [0.2421,0.1802,0.1787],
    [0.1316,0.0680,0.0712],
    [0.0042,0.0056,0.0047]
])

data_mixtal = np.array([
    [0.2597,0.2092,0.2179],
    [0.1144,0.0638,0.0634],
    [0.0378,0.0392,0.0383]
])
horiz_axis = ["OPT-2.7B", "GPT-NEO-2.7B", "GPT-J-6B"]
vert_axis = ["LogRank", "RoBERTa-Large", "Fast-DetectGPT"]

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

sns.light_palette("seagreen", as_cmap=True)


sns.heatmap(data_gpt35, ax=axes[0],vmin=0.0, vmax=0.3, cmap="YlGn_r", annot=True, fmt=".4f", xticklabels=horiz_axis, yticklabels=vert_axis, cbar=False, linecolor='black', linewidth=0.5,annot_kws={'size': 18})
axes[0].set_yticklabels(vert_axis, size = 16,rotation=0)
axes[0].set_xticklabels(horiz_axis, size = 16, rotation=45)
axes[0].set_xlabel("Proxy Model", size=20)
axes[0].set_ylabel("Detector", size=20)
axes[0].set_title("GPT-3.5-Turbo\n", size=22)
axes[0].xaxis.set_ticks_position('top')
axes[0].xaxis.set_label_position('top')


sns.heatmap(data_llama, ax=axes[1], vmin=0.0, vmax=0.3, cmap="YlGn_r", annot=True, fmt=".4f", xticklabels=horiz_axis, yticklabels=vert_axis, cbar=False, linecolor='black', linewidth=0.5,annot_kws={'size': 18})
axes[1].set_yticklabels(vert_axis, size = 16,rotation=0)
axes[1].set_xticklabels(horiz_axis, size = 16, rotation=45)
axes[1].set_xlabel("Proxy Model", size=20)
axes[1].set_ylabel("Detector", size=20)
axes[1].set_title("Llama-3-70B\n", size=22)
axes[1].xaxis.set_ticks_position('top')
axes[1].xaxis.set_label_position('top')

sns.heatmap(data_mixtal, ax=axes[2], vmin=0.0, vmax=0.3, cmap="YlGn_r", annot=True, fmt=".4f", xticklabels=horiz_axis, yticklabels=vert_axis, cbar=True, linecolor='black', linewidth=0.5,annot_kws={'size': 18})
axes[2].set_yticklabels(vert_axis, size = 16,rotation=0)
axes[2].set_xticklabels(horiz_axis, size = 16, rotation=45)
axes[2].set_xlabel("Proxy Model", size=20)
axes[2].set_ylabel("Detector", size=20)
axes[2].set_title("Mixtral-8x7B-Instruct\n", size=22)
axes[2].xaxis.set_ticks_position('top')
axes[2].xaxis.set_label_position('top')


fig.suptitle("XSum Dataset\n", fontsize=26)

for ax in axes:
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)
        
plt.tight_layout()
plt.savefig("xsum_graphic.pdf",bbox_inches="tight")

# SQuAD
data_gpt35 = np.array([
    [0.3100,0.2544,0.2421],
    [0.2013,0.1460,0.1357],
    [0.0155,0.0287,0.0313]
])

data_llama = np.array([
    [0.2957,0.2273,0.2299],
    [0.0459,0.0289,0.0336],
    [0.0031,0.0095,0.0064]
])

data_mixtal = np.array([
    [0.2854,0.2230,0.2192],
    [0.0610,0.0412,0.0374],
    [0.0069,0.0124,0.0102]
])
horiz_axis = ["OPT-2.7B", "GPT-NEO-2.7B", "GPT-J-6B"]
vert_axis = ["LogRank", "RoBERTa-Large", "Fast-DetectGPT"]

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

sns.light_palette("seagreen", as_cmap=True)


sns.heatmap(data_gpt35, ax=axes[0],vmin=0.0, vmax=0.3, cmap="YlGn_r", annot=True, fmt=".4f", xticklabels=horiz_axis, yticklabels=vert_axis, cbar=False, linecolor='black', linewidth=0.5,annot_kws={'size': 18})
axes[0].set_yticklabels(vert_axis, size = 16,rotation=0)
axes[0].set_xticklabels(horiz_axis, size = 16, rotation=45)
axes[0].set_xlabel("Proxy Model", size=20)
axes[0].set_ylabel("Detector", size=20)
axes[0].set_title("GPT-3.5-Turbo\n", size=22)
axes[0].xaxis.set_ticks_position('top')
axes[0].xaxis.set_label_position('top')


sns.heatmap(data_llama, ax=axes[1], vmin=0.0, vmax=0.3, cmap="YlGn_r", annot=True, fmt=".4f", xticklabels=horiz_axis, yticklabels=vert_axis, cbar=False, linecolor='black', linewidth=0.5,annot_kws={'size': 18})
axes[1].set_yticklabels(vert_axis, size = 16,rotation=0)
axes[1].set_xticklabels(horiz_axis, size = 16, rotation=45)
axes[1].set_xlabel("Proxy Model", size=20)
axes[1].set_ylabel("Detector", size=20)
axes[1].set_title("Llama-3-70B\n", size=22)
axes[1].xaxis.set_ticks_position('top')
axes[1].xaxis.set_label_position('top')

sns.heatmap(data_mixtal, ax=axes[2], vmin=0.0, vmax=0.3, cmap="YlGn_r", annot=True, fmt=".4f", xticklabels=horiz_axis, yticklabels=vert_axis, cbar=True, linecolor='black', linewidth=0.5,annot_kws={'size': 18})
axes[2].set_yticklabels(vert_axis, size = 16,rotation=0)
axes[2].set_xticklabels(horiz_axis, size = 16, rotation=45)
axes[2].set_xlabel("Proxy Model", size=20)
axes[2].set_ylabel("Detector", size=20)
axes[2].set_title("Mixtral-8x7B-Instruct\n", size=22)
axes[2].xaxis.set_ticks_position('top')
axes[2].xaxis.set_label_position('top')


fig.suptitle("SQuAD Dataset\n", fontsize=26)

for ax in axes:
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)
        
plt.tight_layout()
plt.savefig("squad_graphic.pdf",bbox_inches="tight")