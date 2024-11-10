# Adapted from https://github.com/shizhouxing/LLM-Detector-Robustness
import os
import json

from utils.attackers import genetic_attack_agent
from utils.model import load_tokenizer, load_model
from utils.openai_perturbations import get_perturbations, save_perturbations, parse_response


class Experiment:
    def __init__(self):
        base_model = load_model("gpt2", "cuda", "../cache")
        base_tokenizer = load_tokenizer("gpt2", "xsum", "../cache")
        self.attacker = genetic_attack_agent(base_tokenizer, base_model, 'cuda')

    def attack_ll(self, idx, text, dataset, attack_method='genetic', attack_model='chatgpt', seed=42):
        random = attack_method == 'random'
        if not os.path.exists('results'):
            os.mkdir('results')
        if not os.path.exists(f'results/chatgpt_seed_{dataset}_{seed}'):
            os.mkdir(f'results/chatgpt_seed_{dataset}_{seed}')

        if not os.path.exists(f'results/chatgpt_seed_{dataset}_{seed}/{idx}.json'):
            results = get_perturbations(text)
            save_perturbations(results, idx, path=f'results/chatgpt_seed_{dataset}_{seed}/{idx}.json')
        mapping = parse_response(f'results/chatgpt_seed_{dataset}_{seed}/{idx}.json')
        for key in mapping.keys():
            if len(mapping[key]) == 1 and '/' in mapping[key][0]:
                mapping[key] = mapping[key][0].split('/')

        if attack_model == 'chatgpt':
            mapping = parse_response(f'results/{attack_model}_seed_{dataset}_{seed}/{idx}.json')
        else:
            raise NotImplementedError()

        for key in mapping.keys():
            if len(mapping[key]) == 1 and '/' in mapping[key][0]:
                mapping[key] = mapping[key][0].split('/')
        if random:
            elite = self.attacker.random_replacement(text, mapping)
        else:
            elite = self.attacker.attack(text, mapping)

        with open(f'results/{attack_method}_results_seed_{seed}_{dataset}/{idx}.json', 'w') as f:
            sampled = ' '.join(elite[0]).replace(' ,', ',').replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ;', ';').replace(' \'', '\'').replace(' ’ ', '\'').replace(' :', ':').replace('<newline>', '\n').replace('`` ', '"').replace(' \'\'', '"').replace('\'\'', '"').replace('.. ', '... ').replace(' )', ')').replace('( ', '(').replace(' n\'t', 'n\'t').replace(' i ', ' I ').replace(' i\'', ' I\'').replace('\\\'', '\'').replace('\n ', '\n').strip()
            output = {"original": text, "sampled": sampled}
            json.dump(output, f)

    def run(self, dataset):
        with open(f"exp_gpt3to4/data/{dataset}_gpt-3.5-turbo.raw_data.json", 'r') as f:
            texts = json.load(f)["sampled"]
        for i in range(200):
            self.attack_ll(i, texts[i], dataset)

if __name__ == "__main__":
    exp = Experiment()
    exp.run("squad")

