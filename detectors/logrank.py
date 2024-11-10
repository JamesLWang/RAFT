# from .detector import Detector

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

def get_rank(logits, labels, device):
    labels = labels.unsqueeze(0).to(device)
    # print(labels.shape)
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1 # convert to 1-indexed rank
    torch.cuda.empty_cache()
    return -ranks.mean().item()

class LogRankDetector:
    def __init__(self,
                 model_name='EleutherAI/gpt-neo-2.7B',
                 device='cuda'):

        self.device = device
        self.precision = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.precision).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        print("%s detector model loaded on %s" % (model_name, device))

    def llm_likelihood(self, query: str, indexes=None):
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        labels = torch.tensor(self.tokenizer.encode(query))
        torch.cuda.empty_cache()
        return get_rank(logits,labels, self.device)

    def crit(self, query: str, indexes=None):
        return self.llm_likelihood(query, indexes=indexes)

# detector = LogRankDetector()
# original_text = "A little girl helps her neighbor overcome his vow of silence he made after his wife passed away 40 years ago. I've lived next door to Mr. Reynolds for as long as I can remember, but I never truly knew him until now. It all started when my daughter, Lily, noticed Mr. Reynolds sitting alone on his porch, staring into the distance. Curiosity piqued, she decided to introduce herself and offer him a drawing she had made. Day after day, she would bring him a new creation, her vibrant imagination bridging the gap between two seemingly disconnected souls. Slowly but surely, a smile started to appear on Mr. Reynolds' face, and his eyes began to regain their spark. Lily's innocence and purity of heart gently nudged him out of his self-imposed isolation. The silence that had shrouded him for decades began to dissolve. Through her persistent acts of kindness, Lily not only helped Mr. Reynolds find his voice again, but she also taught"
# print(detector.llm_likelihood(original_text))
# sampled_text = "A much boy aims hers neighbors surmount His pledging of silent him make After him husband pass down 40 months later ive lived next door to Mr. Reynolds for well long as I can remember, but I never indeed figured him until now. It all started when I daughter, Lily, noticed Mr. Reynolds sitting stand on he porch, staring into the distance. Curiosity piqued, she decided to introduce herself and offer him a drawing she had making Day after hours she would bring him a new creation, her vibrant imagination bridging the gap between five seemingly disconnected souls. Slowly but surely, a smile started to appear on Mr. Reynolds' face, and his eyes began to regain their spark. Lily's innocence and purity of heart gently nudged him out of his self-imposed isolation. The silence that had shrouded him for decades began to dissolve. Through herself persistent acts of kindness, Lily not only helped Mr. Reynolds find his voice again, but she also taught"
# print(detector.llm_likelihood(sampled_text))
