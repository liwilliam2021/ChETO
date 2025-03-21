CASTILIAN_PAIR_PROMPT = """
You are an expert in Spanish linguistics, specializing in regional variations. Your task is to adapt the following Chilean Spanish text into Castilian (Peninsular) Spanish. Follow these guidelines:

1. Replace only the specified Chilean regionalisms (vocabulary, expressions, or idiomatic phrases) with their Castilian equivalents.
2. Preserve the original meaning, tone, and context of the text as closely as possible.
3. Make minimal changes—only adjust the Chileanisms, leaving the rest of the text intact.
4. Note: Some of the listed Chileanisms may include false positives (e.g., phrases that are mischaracterized). Only modify them when they clearly function as regionalisms within the context.

Example:
Input: "¿Estái buscando pega porque te quedaste sin pega?"
Chileanisms: 
Pega: Pega, regional vocabulary specific to Chilean Spanish, meaning: Trabajo.
Estái: Estái: Chilean voseo: singular 2nd person verb with voseo ending 'ái'

Correct adaptation: ¿Estás buscando trabajo porque te quedaste sin trabajo?
Incorrect adaptation: ¿Estás buscando trabajo porque te quedaste sin empleo?
(Do not make unnecessary changes like replacing "trabajo" with "empleo" when not specified.)

Text to adapt:
{paragraph}

Chileanisms to replace (with explanations):
{dialect_matches}

Return the paragraph in Castilian Spanish with only the specified adjustments. Your response should be a clean, grammatically correct version of the text. Output only the adapted paragraph—do not include explanations or additional commentary.
"""

SCORE_PAIR_PROMPT ="""
You are a linguistics expert. Your task is to score how meaningfully the following transformation demonstrates regional dialect differences between Chilean Spanish (original) and Castilian Spanish (transformed). Only score dialect differences.

Score from 0 to 10 where:
- 0 = no noticeable dialect difference
- 10 = strong, clear dialectal transformation with meaningful changes
If the sentence is in English or another non Spanish language, assign a score of 0.

Original (Chilean):
{original}

Transformed (Castilian):
{transformed}

Respond ONLY with a single number between 0 and 10.
"""


# TODO: Refine in context learning into distribution
SENTIMENT_PROMPT = """
You are a sentiment analysis model trained to classify text into star ratings from 1 to 5. Given a piece of text in Chilean Spanish, predict the most likely star rating based on sentiment, tone, and content.

Instructions:
Provide only a number from 1 to 5 as the output. Do not write 'Output:' or any other text.
Follow this general guideline:
1 star: Very negative, complaints, strong dissatisfaction.
2 stars: Somewhat negative, issues mentioned, but not completely dissatisfied.
3 stars: Neutral or mixed sentiment, both positive and negative aspects.
4 stars: Mostly positive, minor complaints, but overall satisfied.
5 stars: Very positive, strong recommendation, clear satisfaction.

Example Inputs & Outputs:

Input: "El servicio fue pésimo, la comida llegó fría y tardaron demasiado."
Output: 1

Input: "La comida estaba buena, pero la atención fue lenta."
Output: 3

Input: "Todo excelente, el mejor restaurante que he visitado."
Output: 5

Text to classify:
"""

CONTRASTIVE_PROMPT = """
Given two sentences from the same document, predict if the second sentence is the likely next sentence following the first.

Sentence A: "{anchor}"
Sentence B: "{candidate}"

Question: Is Sentence B likely to directly follow Sentence A in the original text? Respond only with "Yes" or "No".

Answer:
"""

TITLE_PARAGRAPH_PROMPT = """
You are an expert editor. Your task is to determine if a paragraph is relevant to a given title.

Instructions:
- Read the title carefully.
- Read the paragraph carefully.
- Decide if the paragraph is consistent with, or directly related to, the topic or subject described by the title.

Respond with only "Yes" if the paragraph clearly matches the topic of the title.
Respond with only "No" if the paragraph does not match the topic of the title.

Title: "{title}"

Paragraph: "{paragraph}"

Does this paragraph match the topic described by the title? Answer "Yes" or "No" only.
"""