# TopWORDS-Poetry

English | [简体中文](README.zh-CN.md)

TopWORDS-Poetry is an unsupervised word discovery and text segmentation method for classical Chinese poetry in the TopWORDS family. It incorporates poetic meter and common segmentation patterns into Bayesian inference to identify poetry-specific words and segment verse without a predefined vocabulary or annotated training corpus. This repository provides the code, prepared metrical-poem data from the *Complete Tang Poetry*, and the main results from the EMNLP 2023 paper.

## Repository structure

```text
.
├── TopwordsSegMix2/                  # TopWORDS-Poetry algorithm
├── TopwordsSeg/                      # TopWORDS / TopWORDS-Seg baselines
├── utils_seg.py                      # Data and segmentation utilities
├── other_method.py                   # General-purpose segmentation baselines
└── RDA/tang/
    ├── poem_0_config.py              # Local path configuration
    ├── poem_1_process.py             # Corpus preprocessing
    ├── poem_2_get_sent_prior.py      # Segmentation-pattern priors
    ├── poem_3_*.py                    # Model and baseline runs
    ├── poem_5_utils.py               # Evaluation functions
    ├── poem_5_evaluate_*.py          # Paper evaluation scripts
    ├── data/                          # Metrical poems and manual annotations
    └── output/20221215/              # Selected results for Table 1
```

## Data and paper results

| Data file | Purpose |
| --- | --- |
| `RDA/tang/data/text_raw.txt` | Original poem texts from the *Complete Tang Poetry*. |
| `RDA/tang/data/poem_data_57.json` | Filtered metrical poems with five or seven characters per line and four or eight lines per poem; includes poem IDs and line structure. |
| `RDA/tang/data/modify_text57.txt` | Single-line model input, one poetry line per row. |
| `RDA/tang/data/modify_text572.txt` | Double-line model input, two poetry lines per row. |
| `RDA/tang/data/word_candidate.txt`, `word_candidate2.txt` | Candidate words generated from poetry segmentation patterns. |
| `RDA/tang/data/sent_prior.pkl`, `sent_prior2.pkl` | Pattern priors for the single-line and double-line models. |
| `RDA/tang/data/label2/sample_data1000.xlsx` | A stratified sample of 1,000 manually segmented poems for evaluation. |

The full method comparison in Table 1 is saved in `RDA/tang/output/20221215/summary_result_2.xlsx`. The two TopWORDS-Poetry configurations are listed below. Each result directory includes `segmented_text.txt`, `word_dictionary.csv`, and a run log.

| Paper method | Result directory (under `output/20221215/`) | Input unit | Overall segmentation F1 |
| --- | --- | --- | ---: |
| TopWORDS-Poetry | `mix23_single/topwords_seg_0.5_0.001/` | One line | 84.48% |
| TopWORDS-Poetry* | `mix23_double/topwords_seg_0.5_0.001/` | Two lines | 84.71% |

| Comparison method | Saved predictions |
| --- | --- |
| TopWORDS | `tw_single/topwords_seg_inf_inf/segmented_text.txt` |
| TopWORDS-Seg | `tw_seg/topwords_seg_1e-05_0.1/segmented_text.txt` |
| LTP, StanfordNLP, THULAC, PKUSEG, Jieba, RandomPattern, SelectPattern, ChatGPT | Corresponding `.txt` files in `other_method/seg_result/` |

## Running and reproducing results

Run the following commands from the repository root. A Python virtual environment is recommended. The core code requires NumPy, pandas, SciPy, NLTK, tqdm, and openpyxl. Regenerating the comparison methods also requires the corresponding segmentation packages.

```bash
python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy pandas scipy nltk tqdm openpyxl
```

The prepared inputs are included, so the two Poetry configurations can be rerun starting from model estimation. First, set `work_dir` in `RDA/tang/poem_0_config.py` to the absolute path of `RDA/tang` in your checkout; the evaluation code uses this configuration to locate the data. The historical scripts contain absolute paths and parameter sweeps. Before running them, update the two scripts according to the table below so each runs only `kappa_d=0.5` and `kappa_s=0.001` and writes to a new `output/reproduced/` directory.

| Setting | Single-line: `poem_3_topwords_seg_mix2_single_server1212.py` | Double-line: `poem_3_topwords_seg_mix2_double_server1212.py` |
| --- | --- | --- |
| `work_dir` | Absolute path to the repository root | Absolute path to the repository root |
| `output_dir` | `join(work_dir, 'RDA', 'tang', 'output', 'reproduced')` | Same as left |
| `filedir` | `join(output_dir, 'mix23_single')` | `join(output_dir, 'mix23_double')` |
| Input and prior | `modify_text57.txt`, `sent_prior.pkl` | `modify_text572.txt`, `sent_prior2.pkl` |
| `protected_prior_words_dir` | `word_candidate.txt` (as recorded in the saved single-line log) | `word_candidate.txt` |
| `sent_mode` | Default value `1` | `2` |
| Parameter loops | `for kappa_d in [0.5]`, `for kappa_s in [0.001]` | Same as left |
| Create output directory | Add `os.makedirs(filedir, exist_ok=True)` before calling `topwords_seg(...)` | Same as left |

The original double-line script names its `filedir` `mix23_double2`; change it to `mix23_double` as shown above. The original single-line script uses `word_candidate2.txt`, while the saved run log records `word_candidate.txt`. Then run:

```bash
python RDA/tang/poem_3_topwords_seg_mix2_single_server1212.py
python RDA/tang/poem_3_topwords_seg_mix2_double_server1212.py
```

The new segmentation and dictionary results will be in `RDA/tang/output/reproduced/mix23_single/topwords_seg_0.5_0.001/` and `RDA/tang/output/reproduced/mix23_double/topwords_seg_0.5_0.001/`, respectively. Each run estimates a model on the full corpus; the original scripts use 10 processes by default.

Evaluate the newly generated segmentation against the 1,000 manually annotated poems:

```bash
python -c "import json
from pathlib import Path
import pandas as pd
from RDA.tang.poem_5_utils import get_seg_idx_and_result_with_type, get_result
data = Path('RDA/tang/data')
output = Path('RDA/tang/output/reproduced')
poems = json.loads((data / 'poem_data_57.json').read_text(encoding='utf-8'))
labels = pd.read_excel(data / 'label2/sample_data1000.xlsx')
indices, segments, _, _ = get_seg_idx_and_result_with_type(labels, poems)
for name, mode in [('mix23_single', 1), ('mix23_double', 2)]:
    path = output / name / 'topwords_seg_0.5_0.001/segmented_text.txt'
    print(name, get_result(path, indices, segments, mode=mode))"
```

The command prints overall accuracy, precision, recall, and F1. To inspect the saved Table 1 values, open `RDA/tang/output/20221215/summary_result_2.xlsx`. The functions in `RDA/tang/poem_5_evaluate_label2_1_entity.py` cover the full word-discovery and multi-method comparison.

## Citation

Changzai Pan et al. (2023). *TopWORDS-Poetry: Simultaneous Text Segmentation and Word Discovery for Classical Chinese Poetry via Bayesian Inference*. Proceedings of EMNLP 2023.
