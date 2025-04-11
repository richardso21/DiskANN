import os
import json
import warnings
import numpy as np
import argparse
from datasets import Dataset
import torch
from tqdm import tqdm

import diskannpy

from utils import Encoder, load_corpus


class DiskannIndexBuilder:
    r"""
    A tool class used to build a diskann index used in retrieval.
    Assuming e5 retrieval model.
    """

    def __init__(
        self,
        model_path,
        corpus_path,
        save_dir,
        max_length,
        batch_size,
        use_fp16,
        search_mem_max,
        pooling_method=None,
        instruction=None,
        embedding_path=None,
        save_embedding=False,
    ):
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.search_mem_max = search_mem_max
        self.instruction = instruction
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding

        # set instruction for encode
        if self.instruction is not None:
            self.instruction = self.instruction.strip() + " "
            print("Set instruction for encoding:", self.instruction)
        else:
            self.instruction = "passage: "  # assume e5
            warnings.warn(f"Instruction is set to default: {self.instruction}")

        # . config pooling method
        if pooling_method is None:
            try:
                # read pooling method from 1_Pooling/config.json
                pooling_config = json.load(
                    open(os.path.join(self.model_path, "1_Pooling/config.json"))
                )
                for k, v in pooling_config.items():
                    if k.startswith("pooling_mode") and v:
                        pooling_method = k.split("pooling_mode_")[-1]
                        if pooling_method == "mean_tokens":
                            pooling_method = "mean"
                        elif pooling_method == "cls_token":
                            pooling_method = "cls"
                        else:
                            # raise warning: not implemented pooling method
                            warnings.warn(
                                f"Pooling method {pooling_method} is not implemented.",
                                UserWarning,
                            )
                            pooling_method = "mean"
                        break
            except Exception as _:
                print(
                    f"Pooling method not found in {self.model_path}, use default pooling method (mean)."
                )
                # use default pooling method
                pooling_method = "mean"
        else:
            if pooling_method not in ["mean", "cls", "pooler"]:
                raise ValueError(f"Invalid pooling method {pooling_method}.")
        self.pooling_method = pooling_method

        self.gpu_num = torch.cuda.device_count()
        # prepare save dir
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn(
                    "Some files already exists in save dir and may be overwritten.",
                    UserWarning,
                )

        self.index_save_path = os.path.join(
            self.save_dir, f"e5_diskann_{self.search_mem_max}"
        )
        os.makedirs(self.index_save_path, exist_ok=True)

        self.embedding_save_path = os.path.join(self.save_dir, "emb_e5.memmap")

        self.corpus: Dataset = load_corpus(self.corpus_path)

        print("Finish loading...")

    @staticmethod
    def _check_dir(dir_path):
        r"""Check if the dir path exists and if there is content."""

        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def build_index(self):
        r"""Constructing different indexes based on selective retrieval method."""
        self.build_dense_index()

    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        all_embeddings = np.memmap(embedding_path, mode="r", dtype=np.float32).reshape(
            corpus_size, hidden_size
        )
        return all_embeddings

    def _save_embedding(self, all_embeddings):
        memmap = np.memmap(
            self.embedding_save_path,
            shape=all_embeddings.shape,
            mode="w+",
            dtype=all_embeddings.dtype,
        )
        length = all_embeddings.shape[0]
        # add in batch
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(
                range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"
            ):
                j = min(i + save_batch_size, length)
                memmap[i:j] = all_embeddings[i:j]
        else:
            memmap[:] = all_embeddings

    def _encode_all(self):
        all_embeddings = []
        for start_idx in tqdm(
            range(0, len(self.corpus), self.batch_size), desc="Inference Embeddings:"
        ):
            batch_data = self.corpus[start_idx : start_idx + self.batch_size][
                "contents"
            ]
            all_embeddings.append(self.encoder.encode(batch_data))

        all_embeddings = np.vstack(all_embeddings)
        all_embeddings = all_embeddings.astype(np.float32)

        return all_embeddings

    @torch.no_grad()
    def build_dense_index(self):
        """Obtain the representation of documents based on the embedding model(BERT-based) and
        construct a diskann index.
        """

        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")

        # create encoder instance to start projecting documents to embedding space
        self.encoder = Encoder(
            model_path=self.model_path,
            pooling_method=self.pooling_method,  # type: ignore
            max_length=self.max_length,
            use_fp16=self.use_fp16,
            instruction=self.instruction,  # type: ignore
        )
        hidden_size = self.encoder.model.config.hidden_size

        if self.embedding_path is not None:
            corpus_size = len(self.corpus)
            all_embeddings = self._load_embedding(
                self.embedding_path, corpus_size, hidden_size
            )
        else:
            all_embeddings = self._encode_all()
            if self.save_embedding:
                self._save_embedding(all_embeddings)
            del self.corpus

        # build index
        print("Creating index")
        diskannpy.build_disk_index(
            data=all_embeddings,
            distance_metric="l2",
            index_directory=self.index_save_path,
            complexity=100,  # default
            graph_degree=64,  # default
            search_memory_maximum=self.search_mem_max,
            build_memory_maximum=100,  # use up to 100GB RAM for index build
            num_threads=0,  # use all available logical processors
            pq_disk_bytes=0,  # do not perform PQ compression
        )  # type: ignore
        print("Finish!")


def main():
    parser = argparse.ArgumentParser(description="Creating index.")

    # Basic parameters
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--save_dir", default="indexes/", type=str)

    # Parameters for building dense index
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--search_mem_max", type=float, default=0.6)
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--embedding_path", default=None, type=str)
    parser.add_argument("--save_embedding", action="store_true", default=False)

    args = parser.parse_args()

    index_builder = DiskannIndexBuilder(
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        search_mem_max=args.search_mem_max,
        pooling_method=args.pooling_method,
        instruction=args.instruction,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
    )
    index_builder.build_index()


if __name__ == "__main__":
    main()
