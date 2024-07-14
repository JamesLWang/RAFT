import json
import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 4, figsize=(12,3.2), sharey=True)
models = ["likelihood", "logrank", "detect-gpt", "fast-detect-gpt"]

for i, ax in enumerate(axs):
    model = models[i]
    with open(f"data/roberta-large_{model}_xsum_0.1/results.json", 'r') as f:
        data = json.load(f)
        GPT = data["original_crits"]
        attack = data["result_crits"]
    with open(f"data/xsum_gpt-3.5-turbo.{model}_crits.json", 'r') as f:
        human = json.load(f)
    ax.hist(attack, bins=10, alpha=0.5, label='GPT-3.5-Turbo + Attack')
    ax.hist(human, bins=10, alpha=0.5, label='Human')
    ax.hist(GPT, bins=10, alpha=0.5, label='GPT-3.5-Turbo')
    ax.set_xlabel('Score', fontsize=13)
    ax.set_ylabel('Frequency' if i == 0 else '', fontsize=13)

axs[0].set_title('Log Prob')
axs[1].set_title('Log Rank')
axs[2].set_title('DetectGPT')
axs[3].set_title('Fast-DetectGPT')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', ncol=3, fontsize=13)

plt.xticks(fontsize=13)
plt.tight_layout()
plt.subplots_adjust(bottom=0.32)

plt.savefig("crit.pdf")

