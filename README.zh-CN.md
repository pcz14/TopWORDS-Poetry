# TopWORDS-Poetry

[English](README.md) | 简体中文

TopWORDS-Poetry 是 TopWORDS 系列面向古典诗歌的无监督词发现与文本分词方法。它将诗歌的格律和常见切分模式融入贝叶斯推断，在不依赖预先给定词表或标注训练语料的条件下，同时识别诗歌词汇并切分诗句。本仓库提供 EMNLP 2023 论文使用的代码、整理后的《全唐诗》格律诗数据，以及表 1 的主要实验结果。

## 目录结构

```text
.
├── TopwordsSegMix2/                  # TopWORDS-Poetry 算法实现
├── TopwordsSeg/                      # TopWORDS / TopWORDS-Seg 基线实现
├── utils_seg.py                      # 数据与分词结果处理工具
├── other_method.py                   # 通用分词基线
└── RDA/tang/
    ├── poem_0_config.py              # 本地路径配置
    ├── poem_1_process.py             # 语料预处理
    ├── poem_2_get_sent_prior.py      # 句式先验构建
    ├── poem_3_*.py                    # 模型与基线运行脚本
    ├── poem_5_utils.py               # 评估函数
    ├── poem_5_evaluate_*.py          # 论文结果评估
    ├── data/                          # 格律诗语料与人工标注
    └── output/20221215/              # 表 1 对应的精选结果
```

## 数据与论文结果

| 数据文件 | 用途 |
| --- | --- |
| `RDA/tang/data/text_raw.txt` | 《全唐诗》原始诗歌文本。 |
| `RDA/tang/data/poem_data_57.json` | 筛选后的五言、七言、四句或八句格律诗；保留诗歌编号和句子结构。 |
| `RDA/tang/data/modify_text57.txt` | 单句模型输入，每行一条诗句。 |
| `RDA/tang/data/modify_text572.txt` | 双句模型输入，每行两条诗句。 |
| `RDA/tang/data/word_candidate.txt`、`word_candidate2.txt` | 按诗歌切分模式生成的候选词。 |
| `RDA/tang/data/sent_prior.pkl`、`sent_prior2.pkl` | 单句和双句模型使用的句式先验。 |
| `RDA/tang/data/label2/sample_data1000.xlsx` | 分层抽样并人工分词标注的 1,000 首诗，用于评估。 |

论文表 1 的完整方法对比保存在 `RDA/tang/output/20221215/summary_result_2.xlsx`。其中两个 TopWORDS-Poetry 配置如下；结果目录中保留了 `segmented_text.txt`、`word_dictionary.csv` 和运行日志。

| 论文中的方法 | 结果目录（位于 `output/20221215/` 下） | 输入单位 | 总体分词 F1 |
| --- | --- | --- | ---: |
| TopWORDS-Poetry | `mix23_single/topwords_seg_0.5_0.001/` | 单句 | 84.48% |
| TopWORDS-Poetry* | `mix23_double/topwords_seg_0.5_0.001/` | 双句 | 84.71% |

| 对比方法 | 已保存的预测结果 |
| --- | --- |
| TopWORDS | `tw_single/topwords_seg_inf_inf/segmented_text.txt` |
| TopWORDS-Seg | `tw_seg/topwords_seg_1e-05_0.1/segmented_text.txt` |
| LTP、StanfordNLP、THULAC、PKUSEG、Jieba、RandomPattern、SelectPattern、ChatGPT | `other_method/seg_result/` 下对应的 `.txt` 文件 |

## 运行与复现

以下命令均从仓库根目录执行。建议使用 Python 虚拟环境；核心代码依赖 NumPy、pandas、SciPy、NLTK、tqdm 和 openpyxl。若需重新运行对比方法，还需安装各方法对应的分词工具。

```bash
python -m venv .venv
# Linux / macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy pandas scipy nltk tqdm openpyxl
```

仓库已提供处理后的输入文件，因此重跑论文中的两个 Poetry 配置可以从模型运行开始。先把 `RDA/tang/poem_0_config.py` 中的 `work_dir` 改为当前仓库内 `RDA/tang` 的绝对路径，供评估代码读取数据。原实验脚本使用绝对路径并包含参数扫描；运行前按下表修改两个脚本中的配置，使单次运行只使用 `kappa_d=0.5`、`kappa_s=0.001`，并将输出放入新的 `output/reproduced/` 目录。

| 配置项 | 单句：`poem_3_topwords_seg_mix2_single_server1212.py` | 双句：`poem_3_topwords_seg_mix2_double_server1212.py` |
| --- | --- | --- |
| `work_dir` | 仓库根目录的绝对路径 | 仓库根目录的绝对路径 |
| `output_dir` | `join(work_dir, 'RDA', 'tang', 'output', 'reproduced')` | 同左 |
| `filedir` | `join(output_dir, 'mix23_single')` | `join(output_dir, 'mix23_double')` |
| 输入与先验 | `modify_text57.txt`、`sent_prior.pkl` | `modify_text572.txt`、`sent_prior2.pkl` |
| `protected_prior_words_dir` | `word_candidate.txt`（与保留的单句日志一致） | `word_candidate.txt` |
| `sent_mode` | 默认值 `1` | `2` |
| 参数循环 | `for kappa_d in [0.5]`、`for kappa_s in [0.001]` | 同左 |
| 输出目录创建 | 在调用 `topwords_seg(...)` 前加入 `os.makedirs(filedir, exist_ok=True)` | 同左 |

原双句脚本的 `filedir` 名为 `mix23_double2`，需按表改为 `mix23_double`；单句脚本原先使用 `word_candidate2.txt`，表中使用的是所保存运行日志记录的 `word_candidate.txt`。修改后执行：

```bash
python RDA/tang/poem_3_topwords_seg_mix2_single_server1212.py
python RDA/tang/poem_3_topwords_seg_mix2_double_server1212.py
```

新生成的分词和词典结果分别位于 `RDA/tang/output/reproduced/mix23_single/topwords_seg_0.5_0.001/` 与 `RDA/tang/output/reproduced/mix23_double/topwords_seg_0.5_0.001/`。运行是完整语料上的模型估计，原脚本默认使用 10 个进程。

用 1,000 首人工标注诗评估新生成的分词结果：

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

命令输出总体准确率、精确率、召回率和 F1。若只想查看论文表 1 的已保存数值，直接打开 `RDA/tang/output/20221215/summary_result_2.xlsx`。完整词发现和多方法对比使用 `RDA/tang/poem_5_evaluate_label2_1_entity.py` 中的评估函数。

## 引用

Changzai Pan et al. (2023). *TopWORDS-Poetry: Simultaneous Text Segmentation and Word Discovery for Classical Chinese Poetry via Bayesian Inference*. Proceedings of EMNLP 2023.
