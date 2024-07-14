# from .detector import Detector

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

def get_likelihood(logits, labels, device):
    labels = labels.unsqueeze(0).to(device)
    assert logits.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1).to(device)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()

class LogProbDetector:
    def __init__(self,
                 model_name='EleutherAI/gpt-neo-2.7B',
                 device='cuda'):
    
        self.precision = torch.float16 if torch.cuda.is_available() else torch.float32
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.precision).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        print("%s detector model loaded on %s" % (model_name, device))

    def get_tokens(self, query: str):
        # Find the token indexes of a word to make attention mask
        tokens_id = self.tokenizer.encode(query)
        tokens_id = tokens_id[:self.tokenizer.model_max_length - 2]
        tokens_id = [self.tokenizer.bos_token_id] + tokens_id + [self.tokenizer.eos_token_id]
        tokens = self.tokenizer.convert_ids_to_tokens(tokens_id)
        return tokens

    def llm_likelihood(self, query: str, indexes=None):
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        labels = torch.tensor(self.tokenizer.encode(query))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return get_likelihood(logits, labels, self.device)
    
    def crit(self, query: str, indexes=None):
        return self.llm_likelihood(query, indexes=indexes)