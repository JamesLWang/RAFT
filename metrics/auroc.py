from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import os
def AUROC(y_scores, y_gt, plot=False, plot_name="img.png", overwrite=False):
    assert len(y_scores) == len(y_gt), f"Inputs y_scores and y_gt have different lengths. y_score length: {len(y_scores)} y_gt length: {len(y_gt)}"
    auc = roc_auc_score(y_gt, y_scores)
    fpr, tpr, thresholds = roc_curve(y_gt, y_scores)

    if plot:
        plt.plot(fpr, tpr, label=f'AUROC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()

        if os.path.exists(plot_name) and not overwrite:
            raise NameError(f"File with {plot_name} exists. Select new name or toggle 'overwrite=True'")
        plt.savefig(plot_name)

    return auc
