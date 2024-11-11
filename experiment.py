import os
import re
import json
import time
import torch
import uuid
import datetime
import argparse
import numpy as np
import nltk
from tqdm import tqdm
from collections import defaultdict
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from openai import OpenAI

from metrics.auroc import AUROC
from detectors.baselines import Baselines
from detectors.ghostbuster import Ghostbuster
from detectors.detect_gpt import Detect_GPT
from detectors.fast_detect_gpt import Fast_Detect_GPT
from detectors.roberta_gpt2_detector_base import GPT2RobertaDetector

client = OpenAI(api_key='YOUR OPENAI API KEY')

def openai_backoff(**kwargs):
    retries, wait_time = 0, 10
    while retries < 10:
        try:
            return client.chat.completions.create(**kwargs)
        except Exception:
            print(f"Waiting for {wait_time} seconds")
            time.sleep(wait_time)
            wait_time *= 2
            retries += 1

class Experiment:
    def __init__(self, dataset, data_generator_llm, proxy_model, detector, output_path, proxy_model_device, target_detector_device, mask_pct, dataset_dir, device_aux=None):
        self.dataset = dataset # Used
        self.data_generator_llm = data_generator_llm # Used
        self.proxy_model = proxy_model
        self.mask_pct = mask_pct
        self.output_path = output_path
        self.dataset_dir = dataset_dir # Used
        self.detector = detector
        self.target_detector_device = target_detector_device
        self.proxy_model_device = proxy_model_device
        self.device_aux = device_aux

        self.proxy_model_tokenizer = None
        self.proxy_model = None

        # [TODO] Implement as lazy loading
        self.word_vectors = KeyedVectors.load_word2vec_format("./assets/GoogleNews-vectors-negative300.bin.gz", binary=True)#, limit=500000)

    def filter_punctuation(string):
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
    
    def get_top_similar_words(self,word, n=15):
        try:
            similar_words = self.word_vectors.most_similar(word, topn=n)
            return similar_words
        except (KeyError, AttributeError):
            print(f"'{word}' is not in the vocabulary.")
            return []
        
    def generate_text(self, query):
        response = openai_backoff(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": query}]
                    )
        print(response)
        return response.choices[0].message.content

    def predict_words(self, paragraph, top_k):
        query = f"""Given some input paragraph, we have highlighted a word using brackets. List {top_k} alternative words for it that ensure grammar correctness and semantic fluency. Output words only.\n{paragraph}"""
        output = self.generate_text(query)
        predicted_words = re.findall(r'\b[a-zA-Z]+\b', output)

        if(type(predicted_words) == list):
            return predicted_words[:top_k]
        if len(predicted_words) == top_k:
            return predicted_words[:top_k]
        else:
            print(f'OpenAI returned else: {predicted_words}')
            return []
        
    def load_data(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        else:
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)
    
    def flatten(self,lst):
        flattened_list = []
        for item in lst:
            if isinstance(item, list):
                flattened_list.extend(self.flatten(item))
            else:
                flattened_list.append(item)
        return flattened_list
    
    def load_dataset(self) -> None:
        '''
        Loads the dataset from the given directory into object self.data.
        '''
        if self.dataset not in ['xsum','squad', 'abstract']:
            raise ValueError("Selected Dataset is invalid. Valid choices: 'xsum','squad','abstract'")
        if self.data_generator_llm not in ['gpt-3.5-turbo', 'mixtral-8x7B-Instruct', 'llama-3-70b-chat']:
            raise ValueError("Selected Data Generator LLM is invalid. Valid choices: 'gpt-3.5-turbo', 'mixtral-8x7B-Instruct', 'llama-3-70b-chat'")
        
        file_name = os.path.join(self.dataset_dir, self.dataset,f"{self.dataset}_{self.data_generator_llm}.raw_data.json")
        if os.path.exists(file_name):
            self.data = self.load_data(file_name)
            print(f"Dataset {self.dataset} generated with {self.data_generator_llm} loaded successfully!")
        else:
            raise ValueError(f"Data filepath {file_name} does not exist")
        
    def load_proxy_model(self) -> None:
        proxy_model_map = {
            'roberta-base-detector': 'roberta-base',
            'roberta-large-detector': 'roberta-large'
        }

        proxy_model_checkpoint_map = {
            'roberta-base-detector': './assets/detector-base.pt',
            'roberta-large-detector': './assets/detector-large.pt'
        }
        if self.proxy_model in proxy_model_map.keys():
            self.proxy_model = GPT2RobertaDetector(model_name=proxy_model_map[self.proxy_model], device=self.proxy_model_device, checkpoint=proxy_model_checkpoint_map[self.proxy_model])
        elif self.proxy_model == 'gpt2':
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            self.proxy_model_tokenizer = GPT2Tokenizer.from_pretrained(self.proxy_model)
            self.proxy_model = GPT2LMHeadModel.from_pretrained(self.proxy_model)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.proxy_model_tokenizer = AutoTokenizer.from_pretrained(self.proxy_model, torch_dtype=torch.float16)
            self.proxy_model = AutoModelForCausalLM.from_pretrained(self.proxy_model, torch_dtype=torch.float16)
        
        if self.proxy_model_device != 'cpu':
            self.proxy_model.to(self.proxy_model_device)
            print(f"{self.proxy_model} model pushed to {self.proxy_model_device}")
        print(f"{self.proxy_model} model and tokenizer (if applicable) loaded successfully!")

    def load_detector(self) -> None:
        if self.detector == 'dgpt':
            self.detector_model = Detect_GPT("./detectors/*sampling_discrepancy.json",0.3, 1.0, 2, 10, "gpt2-xl", "t5-3b", device0=self.target_detector_device, device1=self.target_detector_device)
        elif self.detector == 'fdgpt':
            self.detector_model = Fast_Detect_GPT("gpt2-xl", "gpt2-xl", "xsum", "./detectors/*sampling_discrepancy.json", self.target_detector_device)
        elif self.detector == 'ghostbuster':
            self.detector_model = Ghostbuster()
        elif self.detector == 'logrank':
            self.detector_model = Baselines("logrank", "gpt-neo-2.7B", device=self.target_detector_device)
        elif self.detector == 'logprob':
            self.detector_model = Baselines("likelihood", "gpt-neo-2.7B", device=self.target_detector_device)
        elif self.detector == 'roberta':
            self.detector_model = GPT2RobertaDetector('roberta-large', self.target_detector_device, './assets/detector-large.pt')

    def create_experiment(self) -> None:
        pass
    
    def raft(self) -> None:
        pass

    def analyze_results(self) -> None:
        pass

    def run(self) -> None:
        self.load_dataset()
        self.load_proxy_model()
        self.load_detector()

        self.create_experiment() # TODO
        self.raft() # TODO
        self.analyze_results() # TODO





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['xsum','squad','abstract'], default='xsum')
    parser.add_argument('--mask_pct', default=0.1)
    parser.add_argument('--top_k', default=15)
    parser.add_argument('--data_generator_llm', choices=['gpt-3.5-turbo','mixtral-8x7B-Instruct', 'llama-3-70b-chat'], default='gpt-3.5-turbo')
    parser.add_argument('--proxy_model', choices=['roberta-base-detector', 'roberta-large-detector', 'gpt2', 'opt-2.7b', 'neo-2.7b', 'gpt-j-6b'], default='roberta-base-detector')
    parser.add_argument('--detector', choices=['logprob','logrank','dgpt','fdgpt','ghostbuster', 'roberta-base','roberta-large'], default='roberta-base')
    parser.add_argument('--output_path', default='./experiments/')
    parser.add_argument('--proxy_model_device', default='cuda')
    parser.add_argument('--target_detector_device', default='cuda')
    parser.add_argument('--dataset_dir', default='./datasets/')
    return parser.parse_args()

print("Done Loading!")

if __name__ == "__main__":
    args = get_args()
    print("Done!")
