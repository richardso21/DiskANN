from typing import List
import datasets
from datasets import Dataset
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import diskannpy

def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0) # type: ignore
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None] # type: ignore
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

def load_model(model_path: str, use_fp16: bool = False):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    return model, tokenizer

def load_corpus(corpus_path: str) -> Dataset:
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train")
    return corpus # type: ignore

class Encoder:
    """
    Encoder class for encoding queries using a specified model.

    Attributes:
        model_path (str): The path to the model.
        pooling_method (str): The method used for pooling.
        max_length (int): The maximum length of the input sequences.
        use_fp16 (bool): Whether to use FP16 precision.

    Methods:
        encode(query_list: List[str], is_query=True) -> np.ndarray:
            Encodes a list of queries into embeddings.
    """

    def __init__(self, model_path, pooling_method, max_length, use_fp16):
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)

    @torch.inference_mode()
    def encode(self, query: str) -> np.ndarray:
        query = f"passage: {query}"

        inputs = self.tokenizer(
            query, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        return query_emb

def search(query):
    encoder = Encoder(
        model_path="intfloat/e5-base-v2",
        pooling_method="mean",
        max_length=180,
        use_fp16=True,
    )
    index = diskannpy.StaticDiskIndex(
        index_directory="indexes/e5_diskann_0.6",
        num_threads=0,
        num_nodes_to_cache=0,
    )
    corpus = load_corpus("../FlashRAG/corpus/wiki18_100w.jsonl")

    emb = encoder.encode(query)
    emb = emb.flatten()

    print("embeddings created")

    res = index.search(emb, k_neighbors=10, complexity=10, beam_width=1)
    i = res.identifiers.tolist()
    d = res.distances.tolist()
    for ii, dd in zip(i, d):
        print(corpus[ii], dd)


def main():
    # search("What is the capital of France?")
    search("What is the capital of Japan?")


if __name__ == "__main__":
    main()
