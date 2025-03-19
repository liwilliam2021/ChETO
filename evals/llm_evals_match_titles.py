import json
import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.prompts import *
from utils.apis import *

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

openai_api = OpenAI_API()
anthropic_api = Anthropic_API()

def compute_title_paragraph_matching_accuracy(llm_api, n_samples=120, model_override=None, save_path_base="title_paragraph_matching_output.json"):
    save_path = "outputs/" + llm_api.api_name + "_" + save_path_base
    topic_data = json.load(open("../scrapers/outputs/fotech_output.json"))
    
    documents = list(topic_data.values())
    total_correct = 0
    output_data = []

    # Flatten paragraphs
    all_paragraphs = []
    for doc in documents:
        title = doc["title"]
        for para in doc["text"]:
            all_paragraphs.append({
                "title": title,
                "paragraph": para
            })

    for i in range(n_samples):
        if i % 20 == 0:
            print(f"Processing sample {i}/{n_samples}")

        # 50/50 split between positive and negative examples
        is_positive = (i % 2 == 0)

        if is_positive:
            # Pick title & paragraph from same document
            doc = random.choice(documents)
            title = doc["title"]
            paragraph = random.choice(doc["text"])
            expected = "yes"
        else:
            # Pick title & paragraph from different documents
            doc1, doc2 = random.sample(documents, 2)
            title = doc1["title"]
            paragraph = random.choice(doc2["text"])
            expected = "no"

        # Ask model
        is_match = get_title_paragraph_match(llm_api, title, paragraph, model_override)
        correct = (is_match and expected == "yes") or (not is_match and expected == "no")
        total_correct += int(correct)

        output_data.append({
            "title": title,
            "paragraph": paragraph,
            "expected": expected,
            "model_output": "yes" if is_match else "no",
            "correct": bool(correct)
        })

    # Save output
    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=2)

    accuracy = total_correct / n_samples
    return accuracy

def get_title_paragraph_match(llm_api, title, paragraph, model_override=None):
    messages = [
        {"role": "user", "content": TITLE_PARAGRAPH_PROMPT.format(title=title, paragraph=paragraph)},
    ]
    res = llm_api(messages, model_override).strip().lower()
    if "yes" in res:
        return True
    elif "no" in res:
        return False
    else:
        print(f"Unclear response: {res}")
        raise ValueError("Invalid model response")


print(compute_title_paragraph_matching_accuracy(anthropic_api))