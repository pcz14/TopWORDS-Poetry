text_raw.txt : 最原始文件，记录了所有49474首诗，每首诗为一行。
poem_raw.xlsx : 原始文件生成的excel表格，记录了诗歌id和诗歌文本
poem_data.json : 诗歌转化为data，以诗歌为单位，记录了诗歌的id，文本，是否与前诗重复，几言，几句，是否整齐(regular)
poem_data_57.json : poem_data中五言或七言诗歌，且句数为4或8
modify_text57.txt : 单句为一行，去掉标点且只包含5/7，句子对应诗句数为4或8的句子
modify_text572.txt : 两句为一行，去掉标点且只包含5/7，句子对应诗句数为4或8的句子
word_candidate.txt : 按照212/221模板提取可能词汇候选
word_cbdb.txt : 从cbdb数据named-entities-for-premodern-chinese-history-research-main中提取的人名、地名、官职词汇
