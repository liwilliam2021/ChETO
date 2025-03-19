import json
import random
from collections import defaultdict

# TODO: what is happening with my imports? too tired to debug right now
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.prompts import *
from utils.apis import *

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

openai_api = OpenAI_API()
anthropic_api = Anthropic_API()

def query_llm(llm_api, anchor, candidate):
    prompt = CONTRASTIVE_PROMPT.format(anchor=anchor, candidate=candidate)
    messages = [
        {"role": "user", "content": prompt},
    ]
    res = llm_api(messages).strip()
    return res.lower() in ["yes", "yes."]

import re
def simple_sent_tokenize(text):
    # Split on `.`, `?`, `!` followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def eval_test_set (llm_api, key='https://www.biobiochile.cl/', n=120):
    sample_threshold = 1000
    sample_max = 5000
    random.seed(42)
    with open('../data/chilean_text.json', 'r') as f:
        chilean_text_data = json.load(f)
    filtered_paragraphs = [
        para for para in chilean_text_data[key] if len(simple_sent_tokenize(para)) >= 3
    ]
    
    if len(filtered_paragraphs) > sample_threshold:
        sampled_paragraphs = random.sample(filtered_paragraphs, min(max(sample_threshold, len(filtered_paragraphs)), sample_max))
    sampled_paragraphs = sampled_paragraphs[:n]

    results = []
    correct = 0
    total = 0
    for j, para in enumerate(sampled_paragraphs):
        if j % 10 == 0:
            print(f"Processing paragraph {j}/{len(sampled_paragraphs)}")
        sentences = simple_sent_tokenize(para)
        if len(sentences) < 3:
            continue
        
        for i in range(len(sentences) - 1):
            anchor = sentences[i]
            positive = sentences[i + 1]

            # Negative: random sentence from same paragraph (not i+1)
            candidates_pool = [s for j, s in enumerate(sentences) if j != i + 1 and j != i]
            if not candidates_pool:
                continue
            negative = random.choice(candidates_pool)

            ### Positive pair ###
            pred_pos = query_llm(llm_api, anchor, positive)
            results.append({
                "source": key,
                "anchor": anchor,
                "candidate": positive,
                "label": 1,
                "prediction": int(pred_pos)
            })
            correct += int(pred_pos == True)
            total += 1

            ### Negative pair ###
            pred_neg = query_llm(llm_api, anchor, negative)
            results.append({
                "source": key,
                "anchor": anchor,
                "candidate": negative,
                "label": 0,
                "prediction": int(pred_neg)
            })
            correct += int(pred_neg == False)
            total += 1

    save_path = "outputs/" + llm_api.api_name + "_" + "llm_pairwise_predictions.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    accuracy = correct / total
    print(f"\n✅ Saved {len(results)} predictions to llm_pairwise_predictions.json")
    print(f"✅ Total accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct predictions)")

eval_test_set(openai_api)