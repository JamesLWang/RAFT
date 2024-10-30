#!/bin/bash

python get_baseline_detection_results.py  --dipper --dataset xsum --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector logprob &
python get_baseline_detection_results.py  --dipper --dataset xsum --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector logrank & 
python get_baseline_detection_results.py  --dipper --dataset xsum --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector fdgpt 

# wait

python get_baseline_detection_results.py  --dipper --dataset squad --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector logprob & 
python get_baseline_detection_results.py  --dipper --dataset squad --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector logrank &
python get_baseline_detection_results.py  --dipper --dataset squad --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector fdgpt

# wait

python get_baseline_detection_results.py  --dipper --dataset abstract --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector logprob &
python get_baseline_detection_results.py  --dipper --dataset abstract --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector logrank &
python get_baseline_detection_results.py  --dipper --dataset abstract --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector fdgpt

python get_baseline_detection_results.py  --dipper --dataset xsum --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector dgpt &
python get_baseline_detection_results.py  --dipper --dataset squad --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector dgpt &
python get_baseline_detection_results.py  --dipper --dataset abstract --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector dgpt

python get_baseline_detection_results.py  --dipper --dataset xsum --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector ghostbuster &
python get_baseline_detection_results.py  --dipper --dataset squad --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector ghostbuster &
python get_baseline_detection_results.py  --dipper --dataset abstract --data_generator_llm gpt-3.5-turbo --device cuda:0 --detector ghostbuster