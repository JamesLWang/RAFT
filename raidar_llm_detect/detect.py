# Adapted from https://github.com/cvlab-columbia/RaidarLLMDetect
import json
import numpy as np
from fuzzywuzzy import fuzz

from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

human_rewrite_dir, GPT_rewrite_dir, GPT_attack_rewrite_dir = None, None, None

with open(human_rewrite_dir, 'r') as f:
    data_human = json.load(f)

with open(GPT_rewrite_dir, 'r') as f:
    data_gpt_davinci = json.load(f)

with open(GPT_attack_rewrite_dir, 'r') as f:
    data_gpt_davinci_attack = json.load(f)

def tokenize_and_normalize(sentence):
    # Tokenization and normalization
    return [word.lower().strip() for word in sentence.split()]

def extract_ngrams(tokens, n):
    # Extract n-grams from the list of tokens
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def common_elements(list1, list2):
    # Find common elements between two lists
    return set(list1) & set(list2)

def calculate_sentence_common(sentence1, sentence2):
    tokens1 = tokenize_and_normalize(sentence1)
    tokens2 = tokenize_and_normalize(sentence2)

    # Find common words
    common_words = common_elements(tokens1, tokens2)

    # Find common n-grams (let's say up to 3-grams for this example)
    common_ngrams = set()

    number_common_hierarchy = [len(list(common_words))]

    for n in range(2, 5):  # 2-grams to 3-grams
        ngrams1 = extract_ngrams(tokens1, n)
        ngrams2 = extract_ngrams(tokens2, n)
        common_ngrams = common_elements(ngrams1, ngrams2)
        number_common_hierarchy.append(len(list(common_ngrams)))

    return number_common_hierarchy

ngram_num = 4
def sum_for_list(a,b):
    return [aa+bb for aa, bb in zip(a,b)]

cutoff_start = 0
cutoff_end = 6000000

def get_data_stat(data_json):
    for idxx, each in enumerate(data_json):

        original = each['input']

        raw = tokenize_and_normalize(each['input'])
        if len(raw)<cutoff_start or len(raw)>cutoff_end:
            continue

        statistic_res = {}
        ratio_fzwz = {}
        all_statistic_res = [0 for i in range(ngram_num)]
        cnt = 0
        whole_combined=''
        for pp in each.keys():
            if pp != 'common_features':
                whole_combined += (' ' + each[pp])

                res = calculate_sentence_common(original, each[pp])
                statistic_res[pp] = res
                all_statistic_res = sum_for_list(all_statistic_res, res)

                ratio_fzwz[pp] = [fuzz.ratio(original, each[pp]), fuzz.token_set_ratio(original, each[pp])]
                cnt += 1

        each['fzwz_features'] = ratio_fzwz
        each['common_features'] = statistic_res
        each['avg_common_features'] = [a/cnt for a in all_statistic_res]

        each['common_features_ori_vs_allcombined'] = calculate_sentence_common(original, whole_combined)

        if idxx == 400:
            break

    return data_json

human = get_data_stat(data_human)
gpt_davinci = get_data_stat(data_gpt_davinci)
gpt_davinci_attack = get_data_stat(data_gpt_davinci_attack)

def classifier(human, gpt_davinci, gpt_davinci_attack):

    def get_feature_vec(input_json):
        all_list = []
        for idxx, each in enumerate(input_json):

            try:
                raw = tokenize_and_normalize(each['input'])
                r_len = len(raw)*1.0
            except:
                import pdb; pdb.set_trace()
            each_data_fea  = []

            if r_len ==0:
                continue
            if len(raw)<cutoff_start or len(raw)>cutoff_end:
                continue

            each_data_fea = [ind_d / r_len for ind_d in each['avg_common_features']]
            for ek in each['common_features'].keys():
                each_data_fea.extend([ind_d / r_len for ind_d in each['common_features'][ek]])

            each_data_fea.extend([ind_d / r_len for ind_d in each['common_features_ori_vs_allcombined']])

            for ek in each['fzwz_features'].keys():
                each_data_fea.extend(each['fzwz_features'][ek])

            all_list.append(np.array(each_data_fea))

            if idxx == 400:
                break

        all_list = np.vstack(all_list)

        return all_list

    human_all = get_feature_vec(human)
    gpt_davinci_all = get_feature_vec(gpt_davinci)
    gpt_davinci_attack_all = get_feature_vec(gpt_davinci_attack)

    # reblanced
    h_train, h_test, yh_train, yh_test = train_test_split(human_all, np.zeros(human_all.shape[0]), test_size=0.2, random_state=42)
    davinci_g_train, davinci_g_test, davinci_yg_train, davinci_yg_test = train_test_split(gpt_davinci_all, np.ones(gpt_davinci_all.shape[0]), test_size=0.2, random_state=42)
    _, davinci_attack_g_test, _, davinci_attack_yg_test = train_test_split(gpt_davinci_attack_all, np.ones(gpt_davinci_attack_all.shape[0]), test_size=0.2, random_state=42)

    X_train = np.concatenate((davinci_g_train, h_train), axis=0)
    y_train = np.concatenate((davinci_yg_train, yh_train), axis=0)
    X_test = np.concatenate((davinci_g_test, h_test), axis=0)
    y_test = np.concatenate((davinci_yg_test, yh_test), axis=0)
    X_attack = np.concatenate((davinci_attack_g_test, h_test), axis=0)
    y_attack = np.concatenate((davinci_attack_yg_test, yh_test), axis=0)

    # Neural network
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_attack = scaler.transform(X_attack)

    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='adam', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_attack_pred = clf.predict(X_attack)

    def get_roc_metrics(real_preds, sample_preds):
        fpr, tpr, _ = roc_curve(real_preds, sample_preds)
        roc_auc = auc(fpr, tpr)
        # mask = (fpr <= 0.01)
        # tpr_at_fpr = np.max(tpr * mask)
        tpr_at_fpr = np.interp(0.01, fpr, tpr)
        # print(f"tpr_at_0.01: {tpr_at_fpr}")
        # print()
        return float(roc_auc), float(tpr_at_fpr)

    roc_auc, detection_rate = get_roc_metrics(y_test, y_pred)

    print("original AUROC: ", roc_auc, "DR: ", detection_rate)

    roc_auc, detection_rate = get_roc_metrics(y_attack, y_attack_pred)

    print("attack AUROC: ", roc_auc, "DR: ", detection_rate)

classifier(human, gpt_davinci, gpt_davinci_attack)

