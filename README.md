# 中文文本情感分析

## 项目简介

随着互联网信息的爆炸式增长，大量用户生成的文本数据（如社交媒体评论、商品评价、新闻文章等）承载了丰富的情感信息。
分析和挖掘这些情感信息对于企业决策、产品优化、舆情监控等具有重要的价值。
中文文本情感分析旨在自动识别和分类文本中的情感极性（例如正面、负面或中性），是自然语言处理领域的重要研究方向。

本项目旨在实现中文文本情感分类任务，通过实验比较不同模型在同一数据集上的表现。采用的模型包括：
- LSTM（长短时记忆网络）
- GRU（门控循环单元）
- CNN（卷积神经网络）
- BERT（基于 Transformer 的预训练语言模型）

实验中，为了保证模型性能的可比性，所有模型均使用相似的数据预处理方式，并基于相同的数据集进行训练和评估。

---


## 使用方法

1. **环境准备**
   - 确保安装了 `Python 3.8+`，并使用以下依赖包：
     - `torch`
     - `transformers`
     - `jieba`
     - `gensim`
     - `sklearn`
     - `matplotlib`

2. **中文停用词准备**
   - 使用hit_stopwords.txt来源于[哈工大中文停用词](https://github.com/goto456/stopwords)

3. **数据准备**
   - 在 `./data/` 目录下存放 `train.tsv` 和 `test.tsv` 文件。
   - 数据来源与 [chinese_text_cnn](https://github.com/PracticingMan/chinese_text_cnn?tab=readme-ov-file)
   - 总共有70000条数据，我们将训练数据划分为训练集和验证集，比例为8:2。
   - 数据格式如下：
     - 第一列：标签（1代表正面，0代表负面）
     - 第二列：文本
   - 数据预处理：运行 `data_preprocess.py`进行数据预处理


4. **训练模型**
   - LSTM 模型：运行 `LSTM_main.py`
   - GRU 模型：运行 `GRU_main.py`
   - CNN 模型：运行 `CNN_main.py`
   - BERT 模型：基于Hugging Face上的 [bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)进行训练，运行前需要将模型下载到本地
                运行 `BERT_main.py`

5. **模型测试**
   - 运行 `test.py` 、 `test_bert.py` 可以对模型进行测试

6. **模型评估**
   - 每个模型在训练后自动生成分类报告，并保存训练损失变化图。
   - 运行 `eval.py` 、 `eval_bert.py` 可以对模型进行评估



## 结果对比
| 模型   | Accuracy | Precision | Recall |  F1-score |
| ------ | -------- | -------- | -------- | -------- | 
| LSTM   | 0.55    | 0.70    | 0.55      | 0.44      |
| GRU    | 0.54    | 0.71    | 0.54      | 0.43      | 
| CNN    | 0.68    | 0.72    | 0.68      | 0.66      |
| BERT   | 0.95    | 0.96    | 0.95      | 0.95      | 


---

## 总结

不同模型受训练时间，训练规模，数据集大小的影响，表现各有不同。但也可以看出，BERT模型在中文文本情感分类任务上表现最好。这有多方面因素的影响，但也可以认为BERT模型是当前最好的模型之一
同时，由于数据集较小，模型性能还有很大的提升空间。
此外，详细细节可以看我们的汇报。


