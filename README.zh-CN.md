# TopWORDS-Poetry

[English](README.md) | 简体中文

本仓库收录论文 **TopWORDS-Poetry: Simultaneous Text Segmentation and Word Discovery for Classical Chinese Poetry via Bayesian Inference**（EMNLP 2023）的研究代码、整理后的数据和精选实验结果。

仓库从原项目的 `RDA/tang` 实验文件中筛选而来，保留原始脚本和部分运行结果。这不是一次重新运行的实验，也不是能够确认与论文实验时完全一致的历史源码快照。

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| `TopwordsSegMix2/` | 诗歌实验脚本调用的 TopWORDS-Poetry 算法实现。 |
| `TopwordsSeg/` | TopWORDS 和 TopWORDS-Seg 基线实现。 |
| `RDA/tang/poem_*.py` | 与表 1 有关的数据处理、模型运行、基线生成和评估脚本。 |
| `RDA/tang/data/` | 整理后的《全唐诗》语料、句式先验、候选词，以及 1,000 首诗的人工分词样本。 |
| `RDA/tang/output/20221215/` | 表 1 对应的精选预测结果和原始汇总工作簿。 |

论文表 1 的主要结果位于 `RDA/tang/output/20221215/summary_result_2.xlsx`。其中两个 Poetry 结果对应：

| 论文中的方法 | 实验目录 | 设置 |
| --- | --- | --- |
| TopWORDS-Poetry | `mix23_single/topwords_seg_0.5_0.001/` | 每次处理一句诗；`kappa_d=0.5`、`kappa_s=0.001`。 |
| TopWORDS-Poetry* | `mix23_double/topwords_seg_0.5_0.001/` | 每次处理两句诗；`kappa` 参数相同。 |

工作簿还包含 LTP、StanfordNLP、THULAC、PKUSEG、Jieba、TopWORDS、TopWORDS-Seg、RandomPattern、SelectPattern 和 ChatGPT 的对比结果。仓库中的 `segmented_text.txt` 及基线文本文件是相应的精选分词预测；`word_dictionary.csv` 用于保存词发现结果。完整参数扫描、模型 pickle 文件、探索性输出、幻灯片和旧版代码未收录。

## 数据

`RDA/tang/data/text_raw.txt` 是原始数据处理脚本使用的语料。`poem_data_57.json` 是评估使用的格律诗集合；`modify_text57.txt` 和 `modify_text572.txt` 分别是单句和双句实验的输入。`label2/sample_data1000.xlsx` 是人工标注的评估样本。`word_candidate.txt`、`word_candidate2.txt`、`sent_prior.pkl` 和 `sent_prior2.pkl` 是实验脚本使用的已处理输入。

论文报告的格律诗数量为 34,227 首，而保留的 `poem_data_57.json` 有 34,228 条记录。这一条记录的差异尚未查明。评估样本工作簿有 1,000 行数据。本仓库未重新分发第三方 CBDB 原始数据包或外部词典；原始预处理脚本依赖这些材料，因此仅凭本仓库无法从头生成全部已处理输入。

## 代码与复现说明

实验脚本调用 `TopwordsSegMix2.TopwordsSeg.topwords_seg`；评估脚本读取 `RDA/tang/data/label2/sample_data1000.xlsx` 和 `RDA/tang/output/20221215/` 中的结果。原始 `poem_0_config.py` 及服务器脚本包含当时机器上的绝对路径。在新的检出目录中运行前，需要修改这些路径，并从仓库根目录启动，以便导入 `RDA.tang` 和算法包。

收录代码中可见的核心 Python 依赖包括 NumPy、pandas、SciPy、tqdm，以及 openpyxl 等 Excel 读写引擎。生成基线结果还需要对应方法的分词包。历史运行没有留下记录所有依赖版本与随机种子的锁定文件。

结果目录是为论文评估整理的。例如，`20221215/` 下单句结果的日志记录了 `20221212/` 的原始运行路径。收录的评估脚本记录了方法和数据路径，但不能保证当前源码能逐字节重新生成历史工作簿 `summary_result_2.xlsx`。原项目没有 Git 历史，无法证明这些源码文件与实验运行时的版本完全相同。

## 引用

Changzai Pan et al. (2023). *TopWORDS-Poetry: Simultaneous Text Segmentation and Word Discovery for Classical Chinese Poetry via Bayesian Inference*. Proceedings of EMNLP 2023.

公开发布前，请确认语料和人工标注数据的再分发权利，并为仓库选择许可证。本 README 不授予使用许可。
