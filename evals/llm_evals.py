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

def compute_sentiment_accuracy(llm_api, length=120, model_override=None, save_path_base="predictions_output.json"):
    save_path = "outputs/" + llm_api.api_name + "_" + save_path_base
    sentiment_scores = get_sentiment_scores(llm_api, length, model_override)
    total_error = 0
    output_data = []

    for review in sentiment_scores:
        true = review["rating"]
        predicted = review["predicted_rating"]
        error = abs(true - predicted)
        total_error += error
        output_data.append({
            "review": review["review"],
            "true_rating": true,
            "predicted_rating": predicted
        })

    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=2)

    return 1 - total_error / (5 * length)

def get_sentiment_scores(llm_api, length, model_override=None):
    sentiment_data = json.load(open("../scrapers/outputs/labeled_data.json"))

    # Extract a balanced sample of reviews
    grouped_reviews = defaultdict(list)
    for review in sentiment_data:
        grouped_reviews[review["rating"]].append(review)
    sampled_reviews = []
    for rating in grouped_reviews.keys():
        sampled_reviews.extend(
            random.sample(
                grouped_reviews[rating], min(length, len(grouped_reviews[rating]))
            )
        )
    random.shuffle(sampled_reviews)
    for i, review in enumerate(sampled_reviews):
        if i % 50 == 0:
            print(f"Processing review {i}/{length * len(grouped_reviews.keys())}")
        review["predicted_rating"] = get_sentiment(
            llm_api, review["review"], model_override
        )
    return sampled_reviews


def get_sentiment(llm_api, text, model_override=None):
    messages = [
        {"role": "user", "content": SENTIMENT_PROMPT + "\n" + text},
    ]
    res = llm_api(messages, model_override).strip()
    if not res.isdigit():  # Careful of negatives
        print(res)
        raise ValueError("Invalid response")
    else:
        return int(res)


# TODO: run in script
# print(get_sentiment(anthropic_api, "El servicio fue pésimo, la comida llegó fría y tardaron demasiado."))
print(compute_sentiment_accuracy(anthropic_api))
