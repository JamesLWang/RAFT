# RAFT: Realistic Attacks to Fool Text Detectors

## This repo. is still under construction. Please check back later.

Code from the paper [RAFT: Realistic Attacks to Fool Text Detectors](https://arxiv.org/abs/2410.03658). 

If you use this code, please consider citing the paper as:

```
@inproceedings{wang2024raft,
  title={RAFT: Realistic Attacks to Fool Text Detectors},
  author={Wang, James and Li, Ran and Yang, Junfeng and Mao, Chengzhi},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={},
  year={2024}
}
```

## Setup

Install Python dependencies:

```
pip install -r requirements.txt
```

## Attacks

### Proxy Tasks
#### Next Token Generation
`python main_experiment.py --embedding_llm [gpt2 | opt-2.7b | neo-2.7b | gpt-j-6b])`


#### LLM Detection
`python experiment.py --task llm_detection `

### Constrained Generation 
We use GPT-3.5-turbo to generate substitute candidates. From the substitution candidates, we choose the one that is part-of-speech consistent with the original text and decreases the LLM detection score against the target detector the most. The target detector can be specified by `--detector` in the command:

`python main_experiment.py --detector [logprob | logrank | fdgpt | ghostbuster]`

### Adversarial Training


## Datasets
We evaluated our framework on three datasets: [XSum](https://aclanthology.org/D18-1206.pdf), [SQuAD](https://aclanthology.org/D16-1264.pdf), and [Abstract](https://arxiv.org/pdf/2401.12970). We used [Bao et al.](https://github.com/baoguangsheng/fast-detect-gpt/tree/main/exp_main/data)'s versions of LLM-generated texts for XSum and SQuAD, and follow the same steps as them and DetectGPT to generate LLM-generated texts for Abstract. All datasets used can be found in the `./datasets` directory.
