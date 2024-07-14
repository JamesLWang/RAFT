import json
import glob
import os
from tqdm import tqdm
import re
import torch
import uuid
import datetime
import argparse
import time
import nltk

from metrics.auroc import AUROC

from collections import defaultdict

from gensim.models import KeyedVectors

from detectors.baselines import Baselines
from detectors.ghostbuster import Ghostbuster
from detectors.detect_gpt import Detect_GPT
from detectors.fast_detect_gpt import Fast_Detect_GPT
from detectors.roberta_gpt2_detector import GPT2RobertaDetector

from nltk.tokenize import word_tokenize
from openai import OpenAI


client = OpenAI(api_key='None')


def openai_backoff(**kwargs):
    retries, wait_time = 0, 10
    return client.chat.completions.create(**kwargs)
    while retries < 10:
        try:
            return openai.ChatCompletion.create(**kwargs)
        except:
            print(f"Waiting for {wait_time} seconds")
            time.sleep(wait_time)
            wait_time *= 2
            retries += 1

def generate_text(query):
    response = openai_backoff(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": query}]
                )
    print(response)
    return response.choices[0].message.content

def predict_words(paragraph, top_k):
    query = f"""Given some input paragraph, we have highlighted a word using brackets. List {top_k} alternative words for it that ensure grammar correctness and semantic fluency. Output words only.\n{paragraph}"""
    output = generate_text(query)
    predicted_words = re.findall(r'\b[a-zA-Z]+\b', output)
    
    if(type(predicted_words) == list):
        return predicted_words
    if len(predicted_words) == top_k:
        return predicted_words
    else:
        print(f'OpenAI returned else: {predicted_words}')
        return []

def filter_punctuation(string):
    pattern = r'^[\W_]+|[\W_]+$'

    left_punctuation = re.findall(r'^[\W_]+', string)
    right_punctuation = re.findall(r'[\W_]+$', string)
    clean_string = re.sub(pattern, '', string)

    return ''.join(left_punctuation), ''.join(right_punctuation), clean_string

def get_pos(word):
    tokens = word_tokenize(word)
    tagged = nltk.pos_tag(tokens)
    return tagged[0][1] if tagged else None

def are_same_pos(word1, word2):
    pos1 = get_pos(word1)
    pos2 = get_pos(word2)
    return pos1 == pos2
    
    
def load_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def flatten(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list

class Experiment:    
    def __init__(self, dataset, data_generator_llm, embedding_llm,  output_path, device, mask_pct, dataset_dir, detector, device_aux=None):
        self.dataset = dataset
        self.data_generator_llm = data_generator_llm
        self.embedding_llm = embedding_llm
        self.mask_pct = mask_pct
        self.output_path = output_path
        self.device = device
        self.dataset_dir = dataset_dir
        self.word_vectors = KeyedVectors.load_word2vec_format("./assets/GoogleNews-vectors-negative300.bin.gz", binary=True)#, limit=500000)
        self.detector = args.detector
        self.device_aux = device_aux
        self.detector_model = None
        
    def get_top_similar_words(self,word, n=15):
        try:
            similar_words = self.word_vectors.most_similar(word, topn=n)
            return similar_words
        except (KeyError, AttributeError):
            print(f"'{word}' is not in the vocabulary.")
            return []
        
    def load_dataset(self):
        if self.dataset not in ['xsum','squad','writing','abstract']:
            raise ValueError("Selected Dataset is invalid. Valid choices: 'xsum','squad','writing','abstract'")
        if self.data_generator_llm not in ['gpt-3.5-turbo', 'davinci', 'llama-3-70b-chat', 'mixtral-8x7B-Instruct']:
            raise ValueError("Selected Data Generator LLM is invalid. Valid choices: 'gpt-3.5-turbo', 'davinci'")
        file_name = os.path.join(self.dataset_dir, self.dataset,f"{self.dataset}_{self.data_generator_llm}.raw_data.json")
        if os.path.exists(file_name):
            self.data = load_json_file(file_name)
            print(f"Dataset {self.dataset} generated with {self.data_generator_llm} loaded successfully!")
        else:
            raise ValueError(f"Data filepath {file_name} does not exist")
    
    def load_embedding_llm(self):
        embedding_dict = {
            'gpt2': 'gpt2',
            'opt-2.7b': 'facebook/opt-2.7b',
            'neo-2.7b': 'EleutherAI/gpt-neo-2.7B',
            'gpt-j-6b': 'EleutherAI/gpt-j-6b'
        }
        if self.embedding_llm not in embedding_dict.keys():
            raise ValueError("Selected LLM Embedding not in embedding_dict")
        self.embedding_llm = embedding_dict[self.embedding_llm]
        if self.embedding_llm == 'gpt2':
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            self.embedding_llm_tokenizer = GPT2Tokenizer.from_pretrained(self.embedding_llm)
            self.embedding_llm_model = GPT2LMHeadModel.from_pretrained(self.embedding_llm)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            if self.embedding_llm == 'EleutherAI/gpt-j-6b':
                self.embedding_llm_tokenizer = AutoTokenizer.from_pretrained(self.embedding_llm, torch_dtype=torch.float16)
            else:
                self.embedding_llm_tokenizer = AutoTokenizer.from_pretrained(self.embedding_llm, torch_dtype=torch.float16)
            
            self.embedding_llm_model = AutoModelForCausalLM.from_pretrained(self.embedding_llm, torch_dtype=torch.float16)
            
        if self.device != 'cpu':
            self.embedding_llm_model.to(self.device)
            print(f"{self.embedding_llm} model pushed to {self.device}")
        
        print(f"{self.embedding_llm} model and tokenizer loaded successfully!")
                                      
    def create_experiment(self):
        current_date = datetime.datetime.now()
        formatted_date = current_date.strftime("%Y-%m-%d")

        self.experiment_name = f"{formatted_date}_{self.dataset}_{self.data_generator_llm}_{self.embedding_llm.replace('/','')}_{self.detector}_{str(uuid.uuid4()).split('-')[0]}"
        self.experiment_path = os.path.join(self.output_path, self.experiment_name)
        os.makedirs(self.experiment_path)

        print(f"Experiment Name: {self.experiment_name}")
        print(f"Saving Experiment information to {self.experiment_path}")
        
        self.config = {
            "dataset": self.dataset,
            "data_generator_llm": self.data_generator_llm,
            "embedding_llm": self.embedding_llm,
            "detector": self.detector,
            "mask_pct": self.mask_pct,
            "timestamp_created": str(current_date)
        }
        
        with open(os.path.join(self.experiment_path, 'config.json'), 'w') as f:
            json.dump(self.config, f)
            
        self.res_list = []
        self.result_stats, self.original_stats = defaultdict(list),defaultdict(list)
        print("Experiment setup successfully. Begin data generation")
    
    
    def load_detector(self):
        detector_cuda = self.device
        detector_cuda_aux = self.device_aux
        if self.detector == 'dgpt':
            self.detector_model = Detect_GPT("./detectors/*sampling_discrepancy.json",0.3, 1.0, 2, 10, "gpt2-xl", "t5-3b", device0=detector_cuda_aux, device1=detector_cuda_aux)
        elif self.detector == 'fdgpt': 
            self.detector_model = Fast_Detect_GPT("gpt2-xl", "gpt2-xl", "xsum", "./detectors/*sampling_discrepancy.json", detector_cuda_aux)
        elif self.detector == 'ghostbuster':
            self.detector_model = Ghostbuster()
        elif self.detector == 'logrank':
            self.detector_model = Baselines("logrank", "gpt-neo-2.7B")
        elif self.detector == 'logprob':
            self.detector_model = Baselines("likelihood", "gpt-neo-2.7B")
        elif self.detector == 'roberta':
            self.detector_model = GPT2RobertaDetector(device=detector_cuda_aux)
        else:
            raise ValueError("Incorrect self.detector value")
            
        print("Done")
        print(f"{self.detector} loaded successfully on {detector_cuda_aux}")
            
    def generate_data(self):
        n_samples = len(self.data['sampled'])
        original, result = [], []
        original_stats, result_stats = [], []
        for index in tqdm(range(n_samples)):
            paragraph = self.data['sampled'][index]
            words = paragraph.split()
            len_paragraph = len(words)
            ranks = {}
            
            tokens_id = self.embedding_llm_tokenizer.encode(paragraph,add_special_tokens=True)
            logits = self.embedding_llm_model(torch.tensor(tokens_id).unsqueeze(0).to(self.device)).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            ranks = [(0,-1,'[HEAD]', 0.0)]
            for i in range(1, len(probs[0])):
                token_id = tokens_id[i]
                ranks.append((i, token_id, self.embedding_llm_tokenizer.convert_ids_to_tokens(token_id), probs[0][i-1][token_id].item())) # append as (token position, token id, token, token_prob)
                
                
            ranks.sort(key=lambda x: x[3])
            percent_masked = self.mask_pct
            num_masks = int(len(probs[0]) * percent_masked)
            ranks_filter = list(filter(lambda x: "Ġ" in x[2], ranks))
            
            ctr = 0
            candidates = []
            while ctr < num_masks:
                token_pos, token_id, token, prob = ranks_filter.pop()
                # similar_words = self.get_top_similar_words(self.embedding_llm_tokenizer.decode(token_id).strip())
                candidates.append((token_pos, token_id, token, prob))
                ctr += 1
                
            changes = 0
            best_words = []
            for candidate in candidates:
                token_pos, token_id, token, prob  = candidate
                word = self.embedding_llm_tokenizer.decode(token_id).strip()
                min_score, best_word = self.detector_model.crit(paragraph), word
                
                word_to_replace = self.embedding_llm_tokenizer.decode(tokens_id[token_pos]).strip()
                # print(f'Word to replace: {word_to_replace}')
                paragraph_query = self.embedding_llm_tokenizer.decode(flatten(tokens_id[:token_pos])) + f'[{self.embedding_llm_tokenizer.decode(tokens_id[token_pos]).strip()}]' + self.embedding_llm_tokenizer.decode(flatten(tokens_id[token_pos+1:]))
                
                similar_words = predict_words(paragraph_query, 15) 
                # print(f'Returned candidate words: {similar_words}')
                for similar_word in similar_words:
                    if are_same_pos(word_to_replace, similar_word):
                        paragraph_temp = self.embedding_llm_tokenizer.decode(flatten(tokens_id[:token_pos])) + ' ' + similar_word + ' ' + self.embedding_llm_tokenizer.decode(flatten(tokens_id[token_pos+1:]))
                        score = self.detector_model.crit(paragraph_temp)
                        # print(score, min_score)
                        if score <= min_score:
                            best_word = similar_word
                            min_score = score
                            # print(f'{word_to_replace} replaced by {similar_word}. New score: {min_score}')
                            changes += 1

                best_words.append(best_word)
                if best_word == word:
                    # print(f'Word not replaced')
                    continue
                else:
                    # print(f'Word {word_to_replace} replaced with {best_word}')
                    old_val = tokens_id[token_pos]
                    tokens_id[token_pos] = self.embedding_llm_tokenizer.encode(' ' + best_word.strip(),add_special_tokens=True)
                    # print(token_pos, token_id, token, prob, similar_words)
                    # print(f"Replaced token at {token_pos} with value of {token} to {best_word[0]}. New token value: {tokens_id[token_pos]} | Old Value: {old_val}")

            print(f"Changes made: {changes}")
            
            original.append(paragraph)
            result.append(self.embedding_llm_tokenizer.decode(flatten(tokens_id)))
            
            

            res_json = {
                "original": original[-1],
                "sampled": result[-1],
                "replacement_keys (token_pos, orig_token_id, orig_token)": [(x[0],x[1],x[2]) for x in candidates],
                "best_words": best_words,
                "detector_name": self.detector,
                "original_detector_likelihood": self.detector_model.crit(original[-1]),
                "sampled_detector_likelihood": self.detector_model.crit(result[-1]),
                "mask_pct": self.mask_pct
            }
            
            self.res_list.append(res_json)

            save_path = os.path.join(self.experiment_path, f"results_{index}_{self.detector}.json")
            with open(save_path, 'w') as output_file:
                json.dump(res_json, output_file)

            # TODO
            result_json = {}

            with open(os.path.join(self.experiment_path, "results.json"), 'w') as result_file:
                json.dump(result_json, result_file)
                
    def get_auroc(self):
        experiment_files = sorted(glob.glob(os.path.join("experiments", self.experiment_name, "results_*.json")))
        print(len(experiment_files))

        original_data, sampled_data = [], []
        for experiment_file in tqdm(experiment_files):
            experiment_file_data = load_json_file(experiment_file)
            original_data.append(experiment_file_data['original_detector_likelihood'])
            sampled_data.append(experiment_file_data['sampled_detector_likelihood'])

        labels = [0] * len(original_data) + [1] * len(sampled_data)
        all_data = original_data + sampled_data

        auroc = {'auroc': AUROC(all_data, labels)}
        save_path = os.path.join(self.experiment_path, "auroc.json")
        with open(save_path, 'w') as output_file:
            json.dump(auroc, output_file)
                
        print(auroc)
    
    def run(self):
        self.load_dataset()
        self.load_embedding_llm()
        self.load_detector()
        self.create_experiment()
        self.generate_data()
        self.get_auroc()
        
def get_args():
    parser = argparse.ArgumentParser(description="A simple command-line argument example.")
    parser.add_argument('--dataset', choices=['squad','xsum','writing','abstract'])
    parser.add_argument('--data_generator_llm', choices=['gpt-3.5-turbo', 'davinci', 'mixtral-8x7B-Instruct', 'llama-3-70b-chat'], default='gpt-3.5-turbo')
    parser.add_argument('--embedding_llm', choices=['gpt2', 'opt-2.7b', 'neo-2.7b', 'gpt-j-6b'])
    parser.add_argument('--output_path', default='./experiments/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--aux_device', default='cuda')
    parser.add_argument('--mask_pct', default=0.1)
    parser.add_argument('--dataset_dir', default='./datasets/')
    parser.add_argument('--detector', choices=['logprob','logrank','dgpt','fdgpt','ghostbuster', 'roberta'])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
    experiment = Experiment(args.dataset, args.data_generator_llm, args.embedding_llm, args.output_path, args.device, args.mask_pct, args.dataset_dir, args.detector, args.aux_device)
    experiment.run()
    
    