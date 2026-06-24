from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

text = """
Artificial Intelligence is transforming industries
by automating tasks and improving decision making.
"""

summary = summarizer(
    text,
    max_length=30,
    min_length=10,
    do_sample=False
)

print(summary[0]['summary_text'])