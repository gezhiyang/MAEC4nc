# MAEC

MAEC reranks entity-alignment candidates by routing each example to LLM agents
that score names, graph neighborhoods, image descriptions, and raw images. A
trainable aggregator fuses those scores with the original retrieval scores.

## Setup

Use Python 3.10 or newer and Bash. Install the dependencies before training or
evaluating.

```bash
python -m pip install -r requirements.txt
```

Fill in the empty `API_KEY` field in your local copy of `src/run.py`. Keep the
field empty in the published source. The default endpoint is DashScope's
OpenAI-compatible API. The text and vision models default to `qwen-plus` and
`qwen-vl-plus`. Pass `--text_model_pool` and `--vision_model_pool` to override
the comma-separated model lists.
Use `--base_url` to set a shared endpoint, or `--text_base_url` and
`--vision_base_url` for separate endpoints. A shared `--api_key` overrides the
local key. Use `--text_api_key` and `--vision_api_key` to override either key.

## Data

Prepare the candidate exports from your upstream entity-alignment pipeline.
The default dataset directories are `data/DBP15K/ja_en/` for DBP15K and
`data/icews_wiki/` for ICEWS–WIKI. Use `--data_dir` for the parent directory and
`--data` for the dataset directory name.

```text
<data_dir>/<data>/
├── candidates/
│   ├── cand              # Evaluation candidates
│   ├── train_cand        # Training candidates
│   ├── neighbors
│   ├── name_dict
│   └── pair_mmea_sims
├── img/                  # Optional <entity-id>.png files
└── <descriptions>.json   # Optional
```

The extensionless files in `candidates/` are UTF-8 JSON. An evaluation or
training candidate file maps each query entity ID to a record such as
`{"ref": 12, "ground_rank": 1, "candidates": [34, 12], "cand_sims": [0.85, 0.72]}`.
`ground_rank` is zero-based. `neighbors` maps IDs to arrays of
`[head, relation, tail]` or `[head, relation, tail, start, end]` records.
`name_dict` contains ID-to-name mappings under `ent` and `rel`, with a `time`
mapping for temporal records. `pair_mmea_sims` maps ID pairs such as
`"12-34"` to scores, using the smaller ID first.

Image descriptions can be supplied as an ID-to-caption JSON mapping. The
dataset defaults are `ja_en_img.json` and `WIKI_img.json`. Use
`--image_desc_path` and `--image_root` to change the locations.
Candidate exports, images, and checkpoints are supplied by the runner.

## Train and evaluate

Edit the settings at the top of `run.sh`, then run it from the repository root.
Set `dataset` to `dbp` or `maea`, and `mode` to `train` or `eval`.

```bash
bash run.sh
```

The checked-in settings evaluate DBP15K. The script writes a timestamped log
under `logs/`. For a different data directory or checkpoint, add `--data_dir`
or `--agg_ckpt` to the Python command in `run.sh`.

Training fits the aggregator over agent scores. Evaluation writes rankings
under `result/<data>/` and resumes from the existing result file by default.
Use `--new_result` to rerun evaluation from scratch. If the aggregator
checkpoint is absent, evaluation uses the model's initial equal logits.
Use a trained checkpoint when reproducing trained-aggregator results.

```bash
python -m src.run --help
```

## Code layout

| Module | Responsibility |
| --- | --- |
| `src/run.py` | Dataset presets, training, and evaluation |
| `src/load.py` | Candidate exports, neighborhoods, and optional image evidence |
| `src/loss.py` | Cross-entropy and hard-negative margin loss |
| `src/model.py` | Evidence agents, routing, and score aggregation |
| `src/utils.py` | Ranking metrics, parsing, device handling, and I/O |
