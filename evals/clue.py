# TODO: Make this a main glue module, make a file per eval, wrap the LLM stuff
def compute_glue_like_score(sentiment_score, contrastive_score, title_match_score):
    """
    Computes a composite score by averaging the 3 task scores.
    Each score is normalized between 0 and 1.
    """
    return (sentiment_score + contrastive_score + title_match_score) / 3

print (
    "GPT score:",
    compute_glue_like_score(0.3417, 0.5792, 0.792)
)
print (
    "Anthropic score:",
    compute_glue_like_score(0.392, 0.6792, 0.891)
)