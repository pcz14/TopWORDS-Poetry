# TopWORDS-Poetry

Research code, prepared data, and selected results for **TopWORDS-Poetry: Simultaneous Text Segmentation and Word Discovery for Classical Chinese Poetry via Bayesian Inference** (EMNLP 2023).

This repository is a curated copy of the experiment files in `RDA/tang`. It preserves the original scripts and selected outputs; it is not a newly rerun experiment or an exact historical source-code snapshot.

## Contents

| Path | Contents |
| --- | --- |
| `TopwordsSegMix2/` | TopWORDS-Poetry algorithm used by the poetry experiment scripts. |
| `TopwordsSeg/` | TopWORDS and TopWORDS-Seg baseline implementation. |
| `RDA/tang/poem_*.py` | Data preparation, model runs, baseline generation, and evaluation scripts relevant to Table 1. |
| `RDA/tang/data/` | Prepared Complete Tang Poetry corpus, sentence priors, candidate words, and the 1,000-poem manual segmentation sample. |
| `RDA/tang/output/20221215/` | Selected Table 1 predictions and the original summary workbook. |

The main paper result is `RDA/tang/output/20221215/summary_result_2.xlsx`. Its two poetry rows correspond to:

| Paper row | Experiment directory | Setting |
| --- | --- | --- |
| TopWORDS-Poetry | `mix23_single/topwords_seg_0.5_0.001/` | One poetry line at a time; `kappa_d=0.5`, `kappa_s=0.001`. |
| TopWORDS-Poetry* | `mix23_double/topwords_seg_0.5_0.001/` | Two poetry lines at a time; same `kappa` values. |

The workbook also contains the LTP, StanfordNLP, THULAC, PKUSEG, Jieba, TopWORDS, TopWORDS-Seg, RandomPattern, SelectPattern, and ChatGPT comparisons. The included `segmented_text.txt` files and baseline text files are the selected predictions. `word_dictionary.csv` is retained for the word discovery results. Full parameter sweeps, model pickle files, exploratory outputs, slide decks, and older code versions are omitted.

## Data

`RDA/tang/data/text_raw.txt` is the source corpus used by the original preparation script. `poem_data_57.json` is the filtered metrical-poem collection used by evaluation; `modify_text57.txt` and `modify_text572.txt` are the one-line and two-line model inputs. `label2/sample_data1000.xlsx` contains the manual evaluation sample. `word_candidate.txt`, `word_candidate2.txt`, `sent_prior.pkl`, and `sent_prior2.pkl` are prepared inputs used by the experiment scripts.

The paper reports 34,227 metrical poems, while the preserved `poem_data_57.json` contains 34,228 records. This one-record difference has not been resolved. The evaluation sample workbook has 1,000 data rows. No third-party CBDB source package or external dictionaries are redistributed here; the original preprocessing script references those sources and cannot regenerate every prepared input from this repository alone.

## Code and reproducibility

The experiment scripts import `TopwordsSegMix2.TopwordsSeg.topwords_seg`; the evaluation scripts read `RDA/tang/data/label2/sample_data1000.xlsx` and `RDA/tang/output/20221215/`. The original `poem_0_config.py` and server scripts contain absolute paths from the original machines. Change those paths for a new checkout and run from the repository root so `RDA.tang` and the algorithm packages are importable.

The core Python dependencies visible in the included code are NumPy, pandas, SciPy, tqdm, and an Excel engine such as openpyxl. Baseline-generation scripts additionally require their named segmentation packages. Dependency versions and random seeds for every historical run were not recorded in a lockfile.

The output folder was assembled for paper evaluation: for example, the single-line result log inside `20221215/` records an original run path under `20221212/`. The included evaluation scripts document the methods and data paths, but the historical `summary_result_2.xlsx` cannot be guaranteed to regenerate byte-for-byte with the current source files. There is no Git history in the source project to prove that these files are identical to the versions used at experiment time.

## Citation

Changzai Pan et al. (2023). *TopWORDS-Poetry: Simultaneous Text Segmentation and Word Discovery for Classical Chinese Poetry via Bayesian Inference*. Proceedings of EMNLP 2023.

Before publishing, confirm the redistribution rights for the corpus and manual annotations and choose a license for this repository. No license is granted by this README.
