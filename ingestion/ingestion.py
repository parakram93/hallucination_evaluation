from datasets import load_dataset
import string
import json
from pathlib import Path
from tokenizers import Tokenizer
from bs4 import BeautifulSoup
import unicodedata

dataset = load_dataset(
    "abisee/cnn_dailymail",
    "3.0.0"
)

def to_lowercase(text:str) ->str:
    return text.lower()

# 2. Remove HTML tags
# ─────────────────────────────────────────
def remove_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text()


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())

def clean_text(text: str, 
               lowercase=True,
               remove_html_tags=True,
             ) -> str:
    
    if remove_html_tags:    text = remove_html(text)
    if lowercase:           text = to_lowercase(text)
    if normalize_whitespace: text = normalize_whitespace(text)
    text = normalize_whitespace(text)
    
    return text.strip()

tokenizer_path = Path("tokenizer/my_bpe.json")

tokenizer = Tokenizer.from_file(str(tokenizer_path))
CHUNK_SIZE = 256
MAX_ARTICLES = 1000

all_chunks = []

for sample in dataset["train"].select(range(MAX_ARTICLES)):
    
    article = sample["article"]
    article_id = sample["id"]
    
    cleaned_article = clean_text(article)
    
    encoding = tokenizer.encode(cleaned_article)
    
    token_ids = encoding.ids #output will be something like token_ids = [12, 45, 67, 89, ..., 1000 tokens]
    
    # Chunk by tokens
    for chunk_idx, start in enumerate(
        range(0, len(token_ids), CHUNK_SIZE)
    ):

        chunk_ids = token_ids[
            start:start + CHUNK_SIZE
        ]

        chunk_text = tokenizer.decode(
            chunk_ids
        )

        chunk_data = {
            "chunk_id": chunk_idx,
            "article_id": article_id,
            "source": "cnn_dailymail",
            "text": chunk_text,
            "num_tokens": len(chunk_ids)
        }

        all_chunks.append(chunk_data)  
        
with open(
    "ingestion/chunks.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_chunks,
        f,
        ensure_ascii=False,
        indent=4
    )

print(
    f"Saved {len(all_chunks)} chunks "
    f"from {MAX_ARTICLES} articles."
)