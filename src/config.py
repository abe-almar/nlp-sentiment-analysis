# Column names used in the dataset
names = ["tweet_id", "entity", "sentiment", "text"]

# Mapping from sentiment labels to integer codes
label_to_int = {
    "Negative": 0,
    "Neutral": 1,
    "Irrelevant": 1,  # "Irrelevant" is grouped with Neutral
    "Positive": 2,
}

idx_to_label = {
    0: "Negative",
    1: "Neutral",  # or "Irrelevant" — choose one
    2: "Positive",
}
