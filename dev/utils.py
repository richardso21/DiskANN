import numpy as np
import datasets
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModel


def numpy_to_bin(array, out_file):
    shape = np.array(array.shape)
    npts, ndims = [i.astype(np.uint32) for i in shape]
    with open(out_file, "wb") as f:
        f.write(npts.tobytes())
        f.write(ndims.tobytes())
        f.write(array.tobytes())


def pooling(
    pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(),  # type: ignore
            0.0,
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]  # type: ignore
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True
    )

    return model, tokenizer


def load_corpus(corpus_path: str) -> Dataset:
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train")
    return corpus  # type: ignore


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

    def __init__(
        self,
        model_path: str,
        pooling_method: str,
        max_length: int,
        use_fp16: bool,
        instruction: str = "query: ",
    ):
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.instruction = instruction

        self.model, self.tokenizer = load_model(
            model_path=model_path, use_fp16=use_fp16
        )

    @torch.inference_mode()
    def encode(self, query_list: list[str] | str) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]
        query_list = [f"{self.instruction}{query}" for query in query_list]

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs["input_ids"].shape[0], 1), dtype=torch.long
            ).to(inputs["input_ids"].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output,
                output.last_hidden_state,
                inputs["attention_mask"],
                self.pooling_method,
            )
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        return query_emb

    @torch.inference_mode()
    def encode_all(self, query_list: list[str]) -> np.ndarray:
        """Alias for self.encode"""
        return self.encode(query_list)
