"""
Runs and populates the cvalues
"""

import json
import random
import concurrent.futures

from cvalue import ChileanDialectRules

# TODO: what is happening with my imports? too tired to debug right now
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.prompts import *
from utils.apis import *

CHILEAN_DATA_PATH = "../data/chilean_text.json"
EXAMPLE_DATA_PATH = "../data/chilean_examples.json"
BEST_EXAMPLES_PATH = "../data/best_chilean_examples.json"
TRANSFORMED_EXAMPLES_PATH = "../data/CVALUE.json"

dialect_model = ChileanDialectRules()


def sample_data(max_samples=100000, seed=42):
    if seed:
        random.seed(seed)
    with open(CHILEAN_DATA_PATH, "r") as f:
        data = json.load(f)
    for key in data.keys():
        paragraphs = data[key]
        sampled_paragraphs = random.sample(
            paragraphs, min(max_samples, len(paragraphs))
        )
        data[key] = sampled_paragraphs
    return data


def compute_chilean_examples(data, output_filename=EXAMPLE_DATA_PATH):
    chilean_examples = {}
    for key in data.keys():
        print("Computing for ", key)
        for i, paragraph in enumerate(data[key]):
            if i % 1000 == 0:
                print(f"Processed {i} paragraphs of {len(data[key])} for {key}...")

            all_matches = dialect_model.run_rules(paragraph)
            score = dialect_model.score_matches(all_matches)
            chilean_examples[paragraph] = {
                "matches": all_matches,
                "score": score,
                "source": key,
            }

    # Save results to a JSON file
    with open(output_filename, "w") as outfile:
        json.dump(chilean_examples, outfile, indent=4)

    return chilean_examples


def find_best_examples(
    data=None,
    n=10000,
    input_filename=EXAMPLE_DATA_PATH,
    output_filename=BEST_EXAMPLES_PATH,
):
    if not data:
        with open(input_filename, "r") as f:
            data = json.load(f)

    # Sort examples by score in descending order
    sorted_examples = sorted(data.items(), key=lambda x: x[1]["score"], reverse=True)

    # Select the top 'n' examples
    best_examples = {k: v for k, v in sorted_examples[:n]}

    # Save best examples to a new JSON file
    with open(output_filename, "w") as outfile:
        json.dump(best_examples, outfile, indent=4)

    print(f"Best {n} examples saved to {output_filename}")
    return best_examples


def transform_to_castilian(paragraph, dialect_matches):
    dialect_matches_str = "\n".join(
        [f"{match[0]}: {match[1]}" for match in dialect_matches]
    )
    prompt = CASTILIAN_PAIR_PROMPT.format(
        paragraph=paragraph, dialect_matches=dialect_matches_str
    )
    # Call the ask_gpt function to get the transformed text
    response = ask_gpt([{"role": "system", "content": prompt}])
    return response


def transform_best_examples(
    best_examples=None,
    input_filename=BEST_EXAMPLES_PATH,
    output_filename=TRANSFORMED_EXAMPLES_PATH,
    max_workers=10,
):
    if not best_examples:
        # Load the best examples
        with open(input_filename, "r") as f:
            best_examples = json.load(f)

    transformed_examples = {}

    # Function to handle the transformation of each example
    def process_example(paragraph, example):
        # Get the dialect matches (Chileanismos)
        dialect_matches = [(m["text"], m["explanation"]) for m in example["matches"]]

        # Transform the paragraph into Castilian Spanish
        transformed_paragraph = transform_to_castilian(paragraph, dialect_matches)

        return {
            "transformed_text": transformed_paragraph,
            "original_text": paragraph,
            "matches": dialect_matches,
            "score": example["score"],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        # Submit tasks for each paragraph
        for paragraph, example in best_examples.items():
            futures.append(executor.submit(process_example, paragraph, example))

        # Collect results from the futures as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            paragraph = result["original_text"]
            transformed_examples[paragraph] = result

    # Save transformed examples to a new JSON file
    with open(output_filename, "w") as outfile:
        json.dump(transformed_examples, outfile, indent=4)

    print(f"Transformed examples saved to {output_filename}")
    return transformed_examples


data = sample_data()
examples = compute_chilean_examples(data)
best_examples = find_best_examples(examples)
transform_best_examples(best_examples)
transform_best_examples()
