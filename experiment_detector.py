import os
import re
import json
import nltk
import time
from openai import OpenAI
import argparse
import numpy as np
from nltk.tokenize import word_tokenize

# from roberta_gpt2_detector_large import GPT2RobertaDetector
from detectors.roberta_gpt2_detector import GPT2RobertaDetector

# import sys
# sys.path.append('../ghostbuster')
# from ghostbuster_test import Ghostbuster

# from baselines_test import Baselines
# from detectors.detect_gpt_test import Detect_GPT
from detectors.fast_detect_gpt import Fast_Detect_GPT
# from metrics import get_roc_metrics, get_precision_recall_metrics


client = OpenAI(api_key='None')

def openai_backoff(**kwargs):
    retries, wait_time = 0, 10
    while retries < 10:
        try:
            return client.chat.completions.create(**kwargs)
        except:
            print(f"Waiting for {wait_time} seconds")
            time.sleep(wait_time)
            wait_time *= 2
            retries += 1

class Experiment:
    def __init__(self):
        self.detector = GPT2RobertaDetector()

        # self.tester = Ghostbuster()
        # self.tester = Baselines("logrank", "gpt-neo-2.7B")
        # self.tester = Detect_GPT("./exp_main/results/*perturbation_100.json", 0.3, 1.0, 2, 10, "gpt2-xl", "t5-3b")
        self.tester = Fast_Detect_GPT("gpt2-xl", "gpt2-xl", "xsum", "exp_main/results/*sampling_discrepancy.json", device="cpu")

    def load_data(self, data_file):
        with open(data_file, "r") as fin:
            data = json.load(fin)
        return data

    def resume(self, path, dataset):
        originals, results, original_crits, result_crits = [], [], [], []
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if filename.startswith(dataset) and filename.endswith('.json'):
                filepath = os.path.join(path, filename)
                with open(filepath, 'r') as fp:
                    data = json.load(fp)
                    originals.append(data["original_llm_likelihood"])
                    results.append(data["sampled_llm_likelihood"])
                    original_crits.append(data["original_crit"])
                    result_crits.append(data["sampled_crit"])
        return originals, results, original_crits, result_crits

    def filter_punctuation(self, string):
        pattern = r'^[\W_]+|[\W_]+$'

        left_punctuation = re.findall(r'^[\W_]+', string)
        right_punctuation = re.findall(r'[\W_]+$', string)
        clean_string = re.sub(pattern, '', string)

        return ''.join(left_punctuation), ''.join(right_punctuation), clean_string

    def get_pos(self, word):
        tokens = word_tokenize(word)
        tagged = nltk.pos_tag(tokens)
        return tagged[0][1] if tagged else None

    def are_same_pos(self, word1, word2):
        pos1 = self.get_pos(word1)
        pos2 = self.get_pos(word2)
        return pos1 == pos2

    def generate_text(self, query):
        response = openai_backoff(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": query}]
                    )
        return response["choices"][0]["message"]["content"]

    def predict_words(self, paragraph, top_k):
        query = f"""Given some input paragraph, we have highlighted a word using brackets. List {top_k} alternative words for it that ensure grammar correctness and semantic fluency. Output words only.\n{paragraph}"""
        output = self.generate_text(query)
        predicted_words = re.findall(r'\b[a-zA-Z]+\b', output)
        if len(predicted_words) == top_k:
            return predicted_words
        else:
            return []

    def run(self, detector, tester, dataset, percent_masked, top_k, resume_from=0):
        data_file = f"exp_gpt3to4/data/{dataset}_gpt-3.5-turbo.raw_data.json"
        path = f"exp_new/{detector}_{tester}_{dataset}_{percent_masked}/"
        if not os.path.exists(path):
            os.mkdir(path)
        data, originals, results, original_crits, result_crits = self.load_data(data_file), [], [], [], []
        import pdb
        pdb.breakpoint()
        n_samples = len(data["sampled"])

        if resume_from != 0:
            originals, results, original_crits, result_crits = self.resume(path, dataset)
        for index in range(resume_from, n_samples):
            paragraph = data["sampled"][index]
            words = paragraph.split()
            len_paragraph = len(words)
            ranks = {}

            for i in range(len_paragraph):
                paragraph_new = ' '.join(words[:i] + ['=', words[i], '='] + words[i+1:])
                tokens = self.detector.get_tokens(paragraph_new)
                if tokens[2] == '=':
                    tokens[2] = "Ġ="
                start_end = [i for i, x in enumerate(tokens) if x == "Ġ="]
                highlight_indexes = [i for i in range(start_end[0], start_end[1]-1)]
                ranks[i] = self.detector.llm_likelihood(paragraph, highlight_indexes)

            sorted_keys = [k for k, v in sorted(ranks.items(), key=lambda item: item[1])]
            mask_keys, num_masks = [], int(len_paragraph * percent_masked)
            for key in sorted_keys:
                if num_masks == 0:
                    break
                left_punctuation, right_punctuation, word_to_replace = self.filter_punctuation(words[key])
                paragraph_query = " ".join(words[:key] + [left_punctuation, f"[{word_to_replace}]", right_punctuation] + words[key+1:])
                predicted_words = self.predict_words(paragraph_query, top_k)
                min_score, word_best, replaced = float('inf'), words[key], False
                for predicted_word in predicted_words:
                    if predicted_word not in ['', ' ', word_to_replace] and self.are_same_pos(word_to_replace, predicted_word):
                        predicted_word = left_punctuation + predicted_word + right_punctuation
                        paragraph_new = ' '.join(words[:key] + [predicted_word] + words[key+1:])
                        score = self.tester.crit(paragraph_new)
                        if score <= min_score:
                            word_best = predicted_word
                            min_score = score
                            replaced = True
                if replaced:
                    num_masks -= 1
                    mask_keys.append(key)
                    words[key] = word_best
            paragraph_final = " ".join(words)
            if tester == "ghostbuster":
                original_crit = self.tester.crit(paragraph)
                original = original_crit
                result_crit = self.tester.crit(paragraph_final)
                result = result_crit
            else:
                original, _, original_crit = self.tester.run(paragraph)
                result, _, result_crit = self.tester.run(paragraph_final)
            originals.append(original)
            results.append(result)
            original_crits.append(original_crit)
            result_crits.append(result_crit)

            output_json = {
                "original": paragraph,
                "sampled": paragraph_final,
                "replacement_keys": mask_keys,
                "original_crit": original_crit,
                "sampled_crit": result_crit,
                "original_llm_likelihood": original,
                "sampled_llm_likelihood": result
            }
            with open(os.path.join(path, f"{dataset}_{index}.json"), 'w') as output_file:
                json.dump(output_json, output_file)
            result_json = {
                "mean_original": np.mean(originals),
                "mean_result": np.mean(results),
                "originals": originals,
                "results": results,
                "original_crits": original_crits,
                "result_crits": result_crits
            }
            with open(os.path.join(path, "results.json"), 'w') as result_file:
                    json.dump(result_json, result_file)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['squad','xsum','abstract'])
    parser.add_argument('--mask_pct', default=0.1)
    parser.add_argument('--top_k', default=15)
    parser.add_argument('--detector', choices=['logprob','logrank','dgpt','fdgpt','ghostbuster', 'roberta'])
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    exp = Experiment()
    exp.run(detector="roberta-base", tester=args.detector, dataset=args.dataset, percent_masked=args.mask_pct, top_k=args.top_k)




