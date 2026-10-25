# MultiHop RAG Optimizer

Hybrid retrieval for multi-hop question answering on HotpotQA. Plain BM25
retrieval is compared with a hybrid that re-ranks BM25's candidates by fusing
them with dense embedding search through Reciprocal Rank Fusion (RRF). The
answers come from Llama 3 8B Instruct, and the hybrid is better at every
context size tested, by up to 2.9 points of exact match.

It was developed and run on Purdue's Scholar GPU cluster, and moved here from the cluster with its code,
data and results; the search index and per-question logs are rebuilt by the scripts, so they stay out of the repo.

## Pipeline

1. **Sparse retrieval** (`1_build_bm25_cache.py`): indexes the 21,998 unique
   Wikipedia snippets that come with the questions in a Pyserini/Lucene BM25
   index, and caches the top 20 passages for each question.
2. **Baseline** (`2_eval_baseline_rag.py`): gives the model the top *n* BM25
   passages.
3. **Hybrid** (`3_eval_hybrid_rrf.py`): re-ranks the same 20 BM25 candidates by
   fusing the BM25 rank with the rank from dense similarity
   (`BAAI/bge-large-en-v1.5`), RRF score `1/(60 + rank_bm25) + 1/(60 + rank_dense)`,
   and gives the model the top *n*.
4. **Error analysis** (`4_error_analysis.py`): compares the two runs question
   by question at *n* = 7.

Both runs use `meta-llama/Meta-Llama-3-8B-Instruct` in bf16 with greedy
decoding, a system prompt that asks for only the answer phrase, and up to 20
new tokens. Answers are scored with the standard HotpotQA exact match (EM) and
token F1, for *n* = 1 to 10 passages.

## Results

1,110 HotpotQA dev questions, all of the "hard" level: 895 bridge and 215
comparison questions.

| Passages (*n*) | BM25 EM | BM25 F1 | Hybrid EM | Hybrid F1 |
|---:|---:|---:|---:|---:|
| 1 | 28.1 | 38.4 | 29.3 | 40.6 |
| 2 | 28.9 | 39.6 | 30.8 | 42.4 |
| 3 | 28.8 | 39.8 | 31.7 | 43.4 |
| 4 | 29.3 | 40.0 | 31.4 | 43.1 |
| 5 | 30.5 | 41.5 | 31.7 | 43.8 |
| 6 | 30.5 | 41.2 | 31.9 | 43.4 |
| 7 | 30.7 | 41.6 | **32.0** | **43.8** |
| 8 | **30.8** | **42.0** | 31.5 | 42.9 |
| 9 | 29.6 | 41.1 | 30.5 | 42.8 |
| 10 | 30.6 | 42.0 | 31.0 | 42.8 |

- The hybrid with 3 passages beats the baseline with any number of passages:
  better ranking does more than more context.
- Both peak around 7–8 passages and then decline as extra passages add noise.
- At *n* = 7, the hybrid gains on both question types: bridge EM 26.6 → 27.6
  and comparison EM 47.9 → 50.2.
- Of the questions whose result changed at *n* = 7, the hybrid fixed 50 (30
  bridge, 20 comparison) and broke 36 (21 bridge, 15 comparison); the cases
  are in `hybrid_error_analysis.json`.

## Running it

Requirements: Python 3.11, a GPU with room for an 8B model in bf16, Java 21+
for Pyserini, and access to the gated Llama 3 model on Hugging Face (log in
with `huggingface-cli login`, or set `HF_TOKEN`).

```bash
pip install -r requirements.txt
```

Then run the steps in order:

```bash
python 1_build_bm25_cache.py      # collection/, indexes/, data/bm25_top20_cache.json
python 2_eval_baseline_rag.py     # results_baseline_summary.json, full_log_baseline_n*.json
python 3_eval_hybrid_rrf.py       # results_hybrid_summary.json, full_log_hybrid_n*.json
python 4_error_analysis.py        # hybrid_error_analysis.json
```

`run_baseline.slurm` and `run_hybrid.slurm` run steps 2 and 3 as Slurm jobs
(one GPU, 4-hour limit each). `HF_HOME` sets where models are cached
(default `./hf_cache`).

## Data

- `data/dev.jsonl`: the 1,110 HotpotQA dev questions, one JSON object per line
  with `_id`, `question`, `answer`, `type` and `level`.
- `data/wiki_snippets_dev.json`: each question mapped to its 20 candidate
  Wikipedia passages, the corpus that step 1 indexes.

The questions come from [HotpotQA](https://hotpotqa.github.io/) (Yang et al.,
2018) and the passages from Wikipedia, both under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); these files are
shared under the same license.
