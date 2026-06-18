"""data_stream.py -- streaming shard builder + IterableDataset (constant RAM)."""
import os
import json
import random
from collections import defaultdict
from typing import Dict, Iterator, List

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset

from config import CFG
from utils import parse_year


def _row_to_chain(row) -> Dict:
    chain = {}
    for stage, col in CFG.stage_columns.items():
        text = row.get(col)
        if text is None or len(str(text).strip()) == 0:
            return None
        chain[stage] = str(text).strip()[: CFG.max_stage_chars]
    chain["id"] = str(row.get(CFG.id_column, ""))
    chain["year"] = parse_year(row) or -1
    cites = row.get(CFG.citation_column)
    chain["citations"] = int(cites) if cites is not None else 0
    chain["field"] = str(row.get(CFG.field_column, "unknown")).strip() or "unknown"
    return chain


def build_shards():
    os.makedirs(CFG.shard_dir, exist_ok=True)
    done = os.path.join(CFG.shard_dir, "_DONE")
    if os.path.exists(done):
        print(f"[data] Shards already built in {CFG.shard_dir}")
        return
    print(f"[data] Streaming {CFG.hf_dataset} ...")
    ds = load_dataset(CFG.hf_dataset, split="train", streaming=CFG.streaming)

    def assign_split(chain):
        if chain["year"] == CFG.test_year:
            return "test"
        if chain["year"] != -1 and chain["year"] <= CFG.train_cutoff_year:
            return "train"
        h = (hash(chain["id"]) % 1000) / 1000.0
        return "test" if h < CFG.random_split_test_frac else "train"

    counts = {"train": 0, "test": 0}
    buffers = {"train": [], "test": []}
    field_counts = defaultdict(int)
    CHUNK = 2000

    def flush(split):
        if not buffers[split]:
            return
        table = pa.Table.from_pylist(buffers[split])
        path = os.path.join(CFG.shard_dir, f"{split}_{counts[split]//CHUNK:05d}.parquet")
        pq.write_table(table, path)
        buffers[split] = []

    for row in tqdm(ds, desc="sharding"):
        chain = _row_to_chain(row)
        if chain is None:
            continue
        split = assign_split(chain)
        if split == "train" and CFG.max_papers_per_field > 0:
            if field_counts[chain["field"]] >= CFG.max_papers_per_field:
                continue
            field_counts[chain["field"]] += 1
        cap = CFG.max_train_examples if split == "train" else CFG.max_test_examples
        if counts[split] >= cap:
            if counts["train"] >= CFG.max_train_examples and \
               counts["test"] >= CFG.max_test_examples:
                break
            continue
        buffers[split].append(chain)
        counts[split] += 1
        if len(buffers[split]) >= CHUNK:
            flush(split)
    flush("train"); flush("test")
    with open(done, "w") as f:
        json.dump(counts, f)
    print(f"[data] Wrote shards: {counts}")


class ChainShardDataset(IterableDataset):
    def __init__(self, split: str):
        self.split = split
        self.files = sorted(
            os.path.join(CFG.shard_dir, f) for f in os.listdir(CFG.shard_dir)
            if f.startswith(split) and f.endswith(".parquet"))

    def _iter_rows(self) -> Iterator[Dict]:
        worker = torch.utils.data.get_worker_info()
        files = self.files if worker is None else self.files[worker.id::worker.num_workers]
        for path in files:
            for row in pq.read_table(path).to_pylist():
                yield row

    def __iter__(self):
        buf = []
        for row in self._iter_rows():
            buf.append(row)
            if len(buf) >= CFG.shuffle_buffer:
                random.shuffle(buf)
                while buf:
                    yield buf.pop()
        random.shuffle(buf)
        while buf:
            yield buf.pop()


def load_test_chains() -> List[Dict]:
    files = sorted(
        os.path.join(CFG.shard_dir, f) for f in os.listdir(CFG.shard_dir)
        if f.startswith("test") and f.endswith(".parquet"))
    chains = []
    for path in files:
        chains.extend(pq.read_table(path).to_pylist())
    print(f"[data] Loaded {len(chains)} test chains")
    return chains


if __name__ == "__main__":
    build_shards()
    test = load_test_chains()
    if test:
        ex = test[0]
        print("\n--- Example chain ---")
        for s in CFG.stages:
            print(f"[{s}] {ex[s][:100]}")
