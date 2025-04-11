import diskannpy

from utils import load_corpus, Encoder


def search(query):
    encoder = Encoder(
        model_path="intfloat/e5-base-v2",
        pooling_method="mean",
        max_length=180,
        use_fp16=True,
        instruction="query: ",
    )
    index = diskannpy.StaticDiskIndex(
        index_directory="indexes/e5_diskann_25.0",
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
