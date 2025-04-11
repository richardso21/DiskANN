import numpy as np
import datasets
from tqdm import tqdm

from utils import numpy_to_bin, Encoder


def main():
    query_dataset = datasets.load_dataset("rajpurkar/squad", split="validation")
    assert isinstance(query_dataset, datasets.Dataset)
    encoder = Encoder(
        model_path="intfloat/e5-base-v2",
        pooling_method="mean",
        max_length=180,
        use_fp16=True,
        instruction="query: ",
    )
    res = []
    batch_size = 1024
    for i in tqdm(range(0, len(query_dataset), batch_size)):
        # encode in batches for speedup
        queries = query_dataset["question"][i : i + batch_size]
        res.append(encoder.encode_all(queries))
    res = np.vstack(res)
    numpy_to_bin(res, "./squad_val_qvecs.bin")


if __name__ == "__main__":
    main()
