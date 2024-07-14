# RAFT: Realistic Attacks to Fool Text Detectors

Code from the paper [RAFT: Realistic Attacks to Fool Text Detectors](https://arxiv.org/abs/2410.03658). 

## Setup

Setup Python environment:

```
conda env create -f environment.yml
```
or
```
pip install -r requirements.txt
```

Create `./assets` directory and download the following files:

Fine-tuned `roberta-base` model (478 MB) for detector-based proxy model:
`wget -P ./assets https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-base.pt`

Fine-tuned `roberta-large` model (1.5 GB) for detector-based proxy model:
`wget -P ./assets https://openaipublic.azureedge.net/gpt-2/detector-models/v1/detector-large.pt`

If you want to use Google News word embeddings for word substitution instead of ChatGPT, download the file `GoogleNews-vectors-negative300.bin.gz` from [here](https://code.google.com/archive/p/word2vec/) and put it in the `./assets` directory.

## Demo

Run `streamlit run demo.py` to see a Streamlit demo of RAFT. You will need to provide your own OpenAI API key.

## Attacks

### Proxy Tasks
#### Using autoregressive next-token generation as proxy model
`python experiment.py --dataset [squad | xsum | abstract] --data_generator_llm [gpt-3.5-turbo | davinci | mixtral-8x7B-Instruct | llama-3-70b-chat] --proxy_model [gpt2 | opt-2.7b | neo-2.7b | gpt-j-6b] --detector [logprob | logrank | dgpt |fdgpt | ghostbuster | roberta-base | roberta-large]`

#### Using Roberta GPT-2 detector as proxy model
`python experiment.py --dataset [squad | xsum | abstract] --data_generator_llm [gpt-3.5-turbo | davinci | mixtral-8x7B-Instruct | llama-3-70b-chat] --proxy_model [roberta-base-detector | roberta-large-detector] --detector [logprob | logrank | dgpt |fdgpt | ghostbuster | roberta-base | roberta-large]`

### Constrained Generation 
We use GPT-3.5-turbo to generate substitute candidates. From the substitution candidates, we choose the one that is part-of-speech consistent with the original text and decreases the LLM detection score against the target detector the most. The target detector can be specified by `--detector` in the command.

### Adversarial Training
We demonstrate that RAFT-generated texts can be effectively used to adversarially train detectors on [Raidar](https://arxiv.org/pdf/2401.12970).   

Run `python raidar_llm_detect/rewrite.py` to perform rewrite for human, AI, and RAFT-perturbed AI texts, then run `python raidar_llm_detect/detect.py` to classify.


## Datasets
We evaluated our framework on three datasets: [XSum](https://aclanthology.org/D18-1206.pdf), [SQuAD](https://aclanthology.org/D16-1264.pdf), and [Abstract](https://arxiv.org/pdf/2401.12970). We used [Bao et al.](https://github.com/baoguangsheng/fast-detect-gpt/tree/main/exp_main/data)'s versions of LLM-generated texts for XSum and SQuAD, and follow the same steps as them and DetectGPT to generate LLM-generated texts for Abstract. All datasets used can be found in the `./datasets` directory.

---

If you find our repo useful, please consider citing it as follows:

```
inproceedings{wang-etal-2024-raft,
    title = "{RAFT}: Realistic Attacks to Fool Text Detectors",
    author = "Wang, James Liyuan  and
      Li, Ran  and
      Yang, Junfeng  and
      Mao, Chengzhi",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.939",
    pages = "16923--16936",
}
```