#!/bin/bash

# Step 1: Ensure spacy is installed
if ! python -c "import spacy" &> /dev/null; then
    echo "spaCy not found. Installing spaCy..."

    echo "Homebrew is doing weird things, so installing with this command"
    python3.12 -m pip install -r requirements.txt --break-system-packages
    # pip install spacy
else
    echo "spaCy is already installed."
fi

# Step 2: Ensure es_core_news_sm model is installed
if ! python -c "import spacy; spacy.load('es_core_news_sm')" &> /dev/null; then
    echo "es_core_news_sm model not found. Downloading..."
    python3.12 -m spacy download es_core_news_sm
    # python -m spacy download es_core_news_sm
else
    echo "es_core_news_sm model is already installed."
fi