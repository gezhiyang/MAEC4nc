# MAEC

MAEC reranks entity-alignment candidates by routing each example to LLM agents
that score names, graph neighborhoods, image descriptions, and raw images. A
trainable aggregator fuses those scores with the original retrieval scores.

## Setup

The commands below were verified with Python 3.12. Install the dependencies
before training or evaluating.

```bash
python -m pip install -r requirements.txt
```

Pass your API key with `--api_key`. The default endpoint is DashScope's
OpenAI-compatible API, with `qwen-plus` for text and `qwen-vl-plus` for vision.
Use `--text_model_pool` and `--vision_model_pool` to supply comma-separated
model lists.
Use `--base_url` to set a shared endpoint, or `--text_base_url` and
`--vision_base_url` for separate endpoints. Use `--text_api_key` and
`--vision_api_key` to override the shared key for each service.

## Data

MAEC takes precomputed candidate exports as input. The `dbp` preset uses
`data/DBP15K/ja_en/` (DBP15K Japanese–English), and the `maea` preset uses
`data/icews_wiki/` (ICEWS–WIKI). To change the location, set `--data_dir` to
the parent directory and `--data` to the dataset directory name.

```text
<data_dir>/<data>/
├── candidates/
│   ├── cand              # Required for evaluation
│   ├── train_cand        # Required for training
│   ├── neighbors
│   ├── name_dict
│   └── pair_mmea_sims
├── img/                  # Optional <entity-id>.png files
└── <descriptions>.json   # Optional
```

Both modes require `neighbors`, `name_dict`, `pair_mmea_sims`, and the
candidate file for the selected mode. All four files are UTF-8 JSON. Override
the candidate filename with `--cand_file` for evaluation or `--trainCand_file`
for training.

- The candidate file maps each query entity ID to a record such as
  `{"ref": 12, "ground_rank": 1, "candidates": [34, 12], "cand_sims": [0.85, 0.72]}`.
  `ref` is the reference entity ID, `ground_rank` is its zero-based rank in the
  upstream ranking, and `cand_sims` contains one score per candidate in the
  same order.
- `neighbors` maps entity IDs to lists of ID tuples stored as JSON arrays.
  Use `[head, relation, tail]` for static records or
  `[head, relation, tail, start, end]` for temporal records.
- `name_dict` contains ID-to-name mappings under `ent` and `rel`. For temporal
  records, include `time` with mappings for the start and end IDs.
- `pair_mmea_sims` maps pairs such as `"12-34"` to scores, with the smaller
  entity ID first. Supply an empty JSON object (`{}`) when pair scores are
  unavailable.

Raw images and image descriptions are optional. Images are looked up as
`<entity-id>.png` under `img/`. Descriptions can be supplied as a JSON map
from entity IDs to captions. By default the `dbp` preset looks for
`ja_en_img.json` and `maea` looks for `WIKI_img.json`, both in the dataset
directory. Use `--image_root` and `--image_desc_path` to change those locations.

## Train and evaluate

From the repository root, set the shell variable `MAEC_API_KEY` to your key
and run the desired command. Replace `dbp` with `maea` for ICEWS–WIKI.

```bash
python -m src.run --dataset dbp --run_mode train --api_key "$MAEC_API_KEY" --agg_ckpt checkpoints/dbp.pt
python -m src.run --dataset dbp --run_mode eval --api_key "$MAEC_API_KEY" --agg_ckpt checkpoints/dbp.pt --new_result
```

The optional Bash wrapper `run.sh` defaults to DBP15K evaluation and writes a
timestamped log under `logs/`. To use the wrapper, add the API arguments to
its Python command.

Training fits the aggregator over agent scores and saves its weights to
`--agg_ckpt`. If the checkpoint file is absent, evaluation uses the initial
aggregator weights. Evaluation preserves the upstream rank for entities whose
`ground_rank` is 29 or greater and reranks the remaining entities. Results
are saved under `result/<data>/`. Remove `--new_result` from the example to
resume from an existing result file.

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
