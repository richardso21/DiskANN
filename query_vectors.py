import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import datasets
from tqdm import tqdm

def numpy_to_bin(array, out_file):
    shape = np.array(array.shape)
    npts, ndims = [i.astype(np.uint32) for i in shape]
    with open(out_file, "wb") as f:
        f.write(npts.tobytes())
        f.write(ndims.tobytes())
        f.write(array.tobytes())

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
    def encode_all(self, query_list: list[str]) -> np.ndarray:
        query_list = [f"query: {query}" for query in query_list]

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        ).to("cuda")
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

def main():
    query_dataset = datasets.load_dataset("rajpurkar/squad", split="validation")
    assert isinstance(query_dataset, datasets.Dataset)
    encoder = Encoder(
        model_path="intfloat/e5-base-v2",
        pooling_method="mean",
        max_length=180,
        use_fp16=True,
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