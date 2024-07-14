import streamlit as st
import time 
import re
import nltk
from nltk.tokenize import word_tokenize
from openai import OpenAI
import torch

from detectors.baselines import Baselines
from detectors.ghostbuster import Ghostbuster
from detectors.detect_gpt import Detect_GPT
from detectors.fast_detect_gpt import Fast_Detect_GPT

from detectors.roberta_gpt2_detector_base import GPT2RobertaDetector as GPT2RobertaDetectorBase
from detectors.roberta_gpt2_detector_large import GPT2RobertaDetector as GPT2RobertaDetectorLarge

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRECISION = torch.float16 if torch.cuda.is_available() else torch.float32

openai_key = ""


top_k = 15

st.markdown("""
<style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 450px;
        max-width: 450px;
    }
</style>
""", unsafe_allow_html=True)

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

def openai_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

    retries, wait_time = 0, 10
    while retries < 10:
        try:
            openai.api_key = openai_key
            return openai.ChatCompletion.create(**kwargs)
        except:
            print(f"Waiting for {wait_time} seconds")
            time.sleep(wait_time)
            wait_time *= 2
            retries += 1

def generate_text(query):
    if len(openai_key) == 0:
        return "dog quickly and beautiful jumped if she under imagine happiness red because loudly they toward"
    response = openai_backoff(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": query}],
                    temperature=0.0
                )
    return response.choices[0].message.content

def predict_words(paragraph, top_k):
    query = f"""Given some input paragraph, we have highlighted a word using brackets. List {top_k} alternative words for it that ensure grammar correctness and semantic fluency. Output words only.\n{paragraph}"""
    output = generate_text(query)
    predicted_words = re.findall(r'\b[a-zA-Z]+\b', output)
    if len(predicted_words) > 0:
        return predicted_words
    else:
        print("RETURNED ELSE")
        return []

def flatten(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list

# embedding_llm_model_dict = {
#     'roberta-base': GPT2RobertaDetectorBase(),
#     'roberta-large': GPT2RobertaDetectorLarge(),
#     'gpt2': GPT2LMHeadModel.from_pretrained('gpt2'),
#     'opt-2.7b': AutoModelForCausalLM.from_pretrained('facebook/opt-2.7b', torch_dtype=torch.float16),
#     'neo-2.7b': AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B', torch_dtype=torch.float16),
#     'gpt-j-6b': AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6b', torch_dtype=torch.float16)
# }

# embedding_llm_tokenizer_dict = {
#     'gpt2': GPT2Tokenizer.from_pretrained('gpt2'),
#     'opt-2.7b': AutoTokenizer.from_pretrained('facebook/opt-2.7b', torch_dtype=torch.float16),
#     'neo-2.7b': AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B', torch_dtype=torch.float16),
#     'gpt-j-6b': AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6b', torch_dtype=torch.float16)
# }

# @st.cache_resource  # Caches the model loading to avoid re-initialization
# def load_embedding_llm_model():
#     print("PLACEHOLDER USED")
#     return GPT2LMHeadModel.from_pretrained('gpt2')

# @st.cache_resource  # Caches the model loading to avoid re-initialization
# def load_embedding_llm_tokenizer():
#     print("PLACEHOLDER USED")
#     return GPT2Tokenizer.from_pretrained('gpt2')

# Initialize and cache the LogRank detector
# with st.spinner("Initializing Application..."):
#     detector_model_logrank = load_logrank_detector()
#     embedding_llm_model = load_embedding_llm_model()
#     embedding_llm_tokenizer = load_embedding_llm_tokenizer()
#     print("LOGRANK DETECTOR LOADED")
# END To delete

@st.cache_resource
def load_gpt2_embedding_llm():
    return {
        'model': GPT2LMHeadModel.from_pretrained('gpt2'),
        'tokenizer': GPT2Tokenizer.from_pretrained('gpt2')
    }

@st.cache_resource
def load_opt_27b_embedding_llm():
    return {
        'model': AutoModelForCausalLM.from_pretrained('facebook/opt-2.7b', torch_dtype=PRECISION),
        'tokenizer': AutoTokenizer.from_pretrained('facebook/opt-2.7b', torch_dtype=PRECISION)
    }

@st.cache_resource
def load_neo_27b_embedding_llm():
    return {
        'model': AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B', torch_dtype=PRECISION),
        'tokenizer': AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B', torch_dtype=PRECISION)
    }

@st.cache_resource  
def load_gpt_j_6b_embedding_llm():
    return {
        'model': AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6b', torch_dtype=PRECISION),
        'tokenizer': AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6b', torch_dtype=PRECISION)
    }

@st.cache_resource
def load_gpt2_detector_base():
    return {
        'model': GPT2RobertaDetectorBase()
    }

@st.cache_resource
def load_gpt2_detector_large():
    return {
        'model': GPT2RobertaDetectorLarge()
    }

def load_embedding_llm_model(embedding_llm_choice):
    if embedding_llm_choice == "roberta-base":
        return load_gpt2_detector_base()['model'], None
    elif embedding_llm_choice == "roberta-large":
        return load_gpt2_detector_large()['model'], None
    elif embedding_llm_choice == "gpt2":
        model = load_gpt2_embedding_llm()
        return model['model'], model['tokenizer']
    elif embedding_llm_choice == "opt-2.7b":
        model = load_opt_27b_embedding_llm()
        return model['model'], model['tokenizer']
    elif embedding_llm_choice == "neo-2.7b":
        model = load_neo_27b_embedding_llm()
        return model['model'], model['tokenizer']
    elif embedding_llm_choice == "gpt-j-6b":
        model = load_gpt_j_6b_embedding_llm()
        return model['model'], model['tokenizer']
    else:
        raise ValueError(f"Invalid embedding LLM choice: {embedding_llm_choice}")

@st.cache_resource
def load_logprob_detector():
    return Baselines("likelihood", "gpt-neo-2.7B")

@st.cache_resource
def load_logrank_detector():
    return Baselines("logrank", "gpt-neo-2.7B")

@st.cache_resource
def load_dgpt_detector():
    return Detect_GPT("./detectors/*sampling_discrepancy.json",0.3, 1.0, 2, 10, "gpt2-xl", "t5-3b", device0='cpu', device1='cpu')

@st.cache_resource
def load_fdgpt_detector():
    return Fast_Detect_GPT("gpt2-xl", "gpt2-xl", "xsum", "./detectors/*sampling_discrepancy.json", 'cpu')

@st.cache_resource
def load_ghostbuster_detector():
    return Ghostbuster()

@st.cache_resource
def load_roberta_detector_base():
    return GPT2RobertaDetectorBase()

@st.cache_resource
def load_roberta_detector_large():
    return GPT2RobertaDetectorLarge()

def load_target_detector(detector_choice):
    if detector_choice == "logprob":
        return load_logprob_detector()
    elif detector_choice == "logrank":
        return load_logrank_detector()
    elif detector_choice == "dgpt":
        return load_dgpt_detector()
    elif detector_choice == "fdgpt":
        return load_fdgpt_detector()
    elif detector_choice == "ghostbuster":
        return load_ghostbuster_detector()
    elif test_detector_choice == "GPT-2 Detector (roberta-base)":
        return load_roberta_detector_base()
    elif test_detector_choice == "GPT-2 Detector (roberta-large)":
        return load_roberta_detector_large()
    else:
        raise ValueError(f"Invalid detector choice: {detector_choice}")

def load_test_detector(test_detector_choice):
    if test_detector_choice == "logprob":
        return load_logprob_detector()
    elif test_detector_choice == "logrank":
        return load_logrank_detector()
    elif test_detector_choice == "dgpt":
        return load_dgpt_detector()
    elif test_detector_choice == "fdgpt":
        return load_fdgpt_detector()
    elif test_detector_choice == "ghostbuster":
        return load_ghostbuster_detector()
    elif test_detector_choice == "GPT-2 Detector (roberta-base)":
        return load_roberta_detector_base()
    elif test_detector_choice == "GPT-2 Detector (roberta-large)":
        return load_roberta_detector_large()
    else:
        raise ValueError(f"Invalid detector choice: {test_detector_choice}")

st.sidebar.title("Configuration")

# `--text` input
text_input = st.sidebar.text_area("Text", "", height=200)

# `--proxy` choice input
proxy_choice = st.sidebar.selectbox("Proxy", ["detection", "generation"])

# Conditional `--embedding_llm` choices based on `proxy_choice`
if proxy_choice == "detection":
    embedding_options = ["roberta-base", "roberta-large"]
else:
    embedding_options = ["gpt2", "opt-2.7b", "neo-2.7b", "gpt-j-6b"]

embedding_llm_choice = st.sidebar.selectbox("Embedding LLM", embedding_options)

# `--detector` choice input
detector_choice = st.sidebar.selectbox(
    "Target Detector", 
    ["logprob", "logrank", "dgpt", "fdgpt", "ghostbuster", "GPT-2 Detector (roberta-base)", "GPT-2 Detector (roberta-large)"]
)

test_detector_choice = st.sidebar.selectbox(
    "Test Detector", 
    ["logprob", "logrank", "dgpt", "fdgpt", "ghostbuster", "GPT-2 Detector (roberta-base)", "GPT-2 Detector (roberta-large)"]
)

# `--mask_pct` slider input
mask_pct = st.sidebar.slider("Mask Percentage", 0.0, 1.0, 0.1)

openai_key = st.sidebar.text_input("OpenAI Key", "")

client = OpenAI(api_key=openai_key)

# Main panel placeholder text before "Run" is pressed
st.write("### _RAFT_ Demo")
st.write("Please configure the settings in the sidebar and press 'Run' to process the inputs.")

if st.sidebar.button("Run"):
    st.write("### RAFT Attack Output")

    with st.expander("User Input Parameters", expanded=True):
        st.write(f"**Text:** {text_input}")
        st.write(f"**Proxy:** {proxy_choice}")
        st.write(f"**Embedding LLM:** {embedding_llm_choice}")
        st.write(f"**Detector:** {detector_choice}")
        st.write(f"**Mask Percentage:** {mask_pct}")

    with st.spinner("Initializing Dependencies..."):
        text_placeholder = st.empty()

        text_placeholder.write(f"Loading {embedding_llm_choice}...")
        embedding_llm_model, embedding_llm_tokenizer = load_embedding_llm_model(embedding_llm_choice)
        
        text_placeholder.write(f"Loading {detector_choice}...")
        target_detector = load_target_detector(detector_choice) # NOT IMPLEMENTED
        
        text_placeholder.write(f"Loading {test_detector_choice}...")
        if test_detector_choice == detector_choice:
            test_detector = target_detector
        else:
            test_detector = load_test_detector(test_detector_choice) # NOT IMPLEMENTED
        

        text_placeholder.empty()
    st.write("### RAFT Attack Output")

    placeholder_baseline = st.empty()
    with st.spinner("Computing Baseline Detection Probability"):
        detection_likelihood_baseline = test_detector.crit(text_input)
        with st.expander("Original Text Detection Baseline", expanded=True):
            st.write(f"**Detector:** {detector_choice}")
            st.write(f"**LLM-Generated Probability:** {detection_likelihood_baseline:.4f}")
    placeholder_baseline.write("### Baseline Detection Probability")

    placeholder_raft_rewrite = st.empty()
    placeholder_raft_rewrite.write("### RAFT Attacked Text")
    with st.spinner(f"**Using Proxy Scoring Model:** {detector_choice} to identify words/tokens replacement rank..."):
        paragraph = text_input
        words = paragraph.split()
        len_paragraph = len(words)
        ranks = {}

        if proxy_choice == "detection":    
            words_original = words.copy()
            for i in range(len_paragraph):
                paragraph_new = ' '.join(words[:i] + ['=', words[i], '='] + words[i+1:])
                tokens = embedding_llm_model.get_tokens(paragraph_new)
                if tokens[2] == '=':
                    tokens[2] = "Ġ="
                start_end = [i for i, x in enumerate(tokens) if x == "Ġ="]
                highlight_indexes = [i for i in range(start_end[0], start_end[1]-1)]
                ranks[i] = embedding_llm_model.llm_likelihood(paragraph, highlight_indexes)

            sorted_keys = [k for k, v in sorted(ranks.items(), key=lambda item: item[1])]
            sorted_words = [words[k] for k in sorted_keys]
            mask_keys, num_masks = [], int(len_paragraph * mask_pct)
            st.write("#### Proxy Scoring Model Ranking")
            st.write(f"**Proxy Task:** {proxy_choice}")
            st.write(f"**Proxy Task Model:** {embedding_llm_choice}")
            st.write(f"""Words sorted in ranked order (first word is most likely to be replaced): \n\n `{', '.join(sorted_words)}`""")
            st.write(f"""Word index sorted in ranked order (first index is most likely to be replaced): \n\n `{', '.join(map(str, sorted_keys))}`""")
        
        if proxy_choice == "generation":
            tokens_id = embedding_llm_tokenizer.encode(paragraph,add_special_tokens=True)
            logits = embedding_llm_model(torch.tensor(tokens_id).unsqueeze(0).to(DEVICE)).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            ranks = [(0,-1,'[HEAD]', 0.0)]
            for i in range(1, len(probs[0])):
                token_id = tokens_id[i]
                ranks.append((i, token_id, embedding_llm_tokenizer.convert_ids_to_tokens(token_id), probs[0][i-1][token_id].item())) # append as (token position, token id, token, token_prob)
            ranks.sort(key=lambda x: x[3])
            percent_masked = mask_pct
            num_masks = int(len(probs[0]) * percent_masked)
            ranks_filter = list(filter(lambda x: "Ġ" in x[2], ranks))     
            for rank_filter in ranks_filter:
                if rank_filter[2].replace("Ġ", "") == "":
                    ranks_filter.remove(rank_filter)

            sorted_keys = [x[0] for x in ranks_filter]
            sorted_words = [x[2].replace("Ġ", "") for x in ranks_filter]
            st.write("#### Proxy Scoring Model Ranking")
            st.write(f"**Proxy Task:** {proxy_choice}")
            st.write(f"**Proxy Task Model:** {embedding_llm_choice}")
            st.write(f"""Words sorted in ranked order (first word is most likely to be replaced): \n\n `{', '.join(sorted_words)}`""")
            st.write(f"""Word index sorted in ranked order (first index is most likely to be replaced): \n\n `{', '.join(map(str, sorted_keys))}`""")
            mask_keys, num_masks = [], int(len_paragraph * mask_pct)

    with st.spinner(f"Generating repalcement candidates, filtering for POS consistency, and greedily optimizing against target detector {detector_choice}..."):
            if proxy_choice == "detection":
                for key in sorted_keys:
                    if num_masks == 0:
                        break
                    left_punctuation, right_punctuation, word_to_replace = filter_punctuation(words[key])
                    paragraph_query = " ".join(words[:key] + [left_punctuation, f"[{word_to_replace}]", right_punctuation] + words[key+1:])
                    predicted_words = predict_words(paragraph_query, top_k)
                    min_score, word_best, replaced = float('inf'), words[key], False
                    for predicted_word in predicted_words:
                        if predicted_word not in ['', ' ', word_to_replace] and are_same_pos(word_to_replace, predicted_word):
                            predicted_word = left_punctuation + predicted_word + right_punctuation
                            paragraph_new = ' '.join(words[:key] + [predicted_word] + words[key+1:])
                            score = target_detector.crit(paragraph_new)
                            if score <= min_score:
                                word_best = predicted_word
                                min_score = score
                                replaced = True
                    if replaced:
                        num_masks -= 1
                        mask_keys.append(key)
                        words[key] = word_best
                
                
                st.write("**RAFT Attacked Text**")
                st.markdown(f"<span style='background-color: #ffeb3b'>{' '.join(words)}</span>", unsafe_allow_html=True)
                new_text = ' '.join(words)
            
            if proxy_choice == "generation":
                print("TO FIX")
                ctr = 0
                candidates = []
                while ctr < num_masks:
                    token_pos, token_id, token, prob = ranks_filter.pop()                    
                    candidates.append((token_pos, token_id, token, prob))
                    ctr += 1

                changes = 0
                best_words = []

                for candidate in candidates:
                    token_pos, token_id, token, prob  = candidate
                    word = embedding_llm_tokenizer.decode(token_id).strip()
                    min_score, best_word = target_detector.crit(paragraph), word
                    
                    word_to_replace = embedding_llm_tokenizer.decode(tokens_id[token_pos]).strip()
                    paragraph_query = embedding_llm_tokenizer.decode(flatten(tokens_id[:token_pos])) + f'[{embedding_llm_tokenizer.decode(tokens_id[token_pos]).strip()}]' + embedding_llm_tokenizer.decode(flatten(tokens_id[token_pos+1:]))
                    
                    similar_words = predict_words(paragraph_query, 15) 
                    for similar_word in similar_words:
                        if are_same_pos(word_to_replace, similar_word):
                            paragraph_temp = embedding_llm_tokenizer.decode(flatten(tokens_id[:token_pos])) + ' ' + similar_word + ' ' + embedding_llm_tokenizer.decode(flatten(tokens_id[token_pos+1:]))
                            score = target_detector.crit(paragraph_temp)
                            if score <= min_score:
                                best_word = similar_word
                                min_score = score
                                changes += 1

                    best_words.append(best_word)
                    if best_word == word:
                        continue
                    else:
                        old_val = tokens_id[token_pos]
                        tokens_id[token_pos] = embedding_llm_tokenizer.encode(' ' + best_word.strip(),add_special_tokens=True)

                print(f"Changes made: {changes}")
                words = embedding_llm_tokenizer.decode(flatten(tokens_id)).split()
                st.write("**RAFT Attacked Text**")
                st.markdown(f"<span style='background-color: #ffeb3b'>{' '.join(words)}</span>", unsafe_allow_html=True)
                new_text = ' '.join(words)
                words_original = paragraph.split()
            
            
            # Create a diff visualization
            diff_html = ""
            for orig, new in zip(words_original, words):
                if orig != new:
                    diff_html += f'<span style="background-color: #ffd1d1; text-decoration: line-through;">{orig}</span> '
                    diff_html += f'<span style="background-color: #c1ffc1;">{new}</span> '
                else:
                    diff_html += orig + " "
            
            st.write("**Original vs Modified Text:**")
            st.markdown(diff_html, unsafe_allow_html=True)
            
            # Display a summary of changes
            changes = [(orig, new) for orig, new in zip(words_original, words) if orig != new]
            if changes:
                st.write("**Word Replacements:**")
                for orig, new in changes:
                    st.write(f"- '{orig}' → '{new}'")

            

    placeholder_raft_detection = st.empty()
    with st.spinner("Computing RAFT Attack Detection Probability"):
        detection_likelihood_raft = test_detector.crit(new_text)
        with st.expander("RAFT Attack Detection Probability", expanded=True):
            st.write(f"**Detector:** {test_detector_choice}")
            st.write(f"**LLM-Generated Probability:** {detection_likelihood_raft:.4f}")

    original_detection = test_detector.crit(text_input)
    improvement = original_detection - detection_likelihood_raft
    
    if improvement > 0:
        st.markdown(f"**Improvement:** <span style='color: #00aa00'>-{improvement:.4f}</span> (Lower LLM detection likelihood)", unsafe_allow_html=True)
    else:
        st.markdown(f"**Change:** <span style='color: #aa0000'>+{improvement:.4f}</span> (Higher LLM detection likelihood)", unsafe_allow_html=True)
    
    placeholder_raft_detection.write("### RAFT Attack Detection Probability")

                