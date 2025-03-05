import os
from pathlib import Path


def get_contexts(n: int, num_words: int, data_dir: str):
    data_dir = Path(data_dir)
    contexts = []
    for filename in os.listdir(data_dir)[:n]:
        with open(data_dir / filename) as f:
            sample = " ".join(f.readlines()[0].split(" ")[:num_words])
            contexts.append(sample)
    return contexts
