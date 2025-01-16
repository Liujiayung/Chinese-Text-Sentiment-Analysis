import torch
from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np

# 数据加载和预处理函数
def preprocess_text(text, tokenizer, max_length=128):
    encoded = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return encoded['input_ids'], encoded['attention_mask']

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载BERT分词器和模型
    bert_path = './bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert_model = BertModel.from_pretrained(bert_path, ignore_mismatched_sizes=True).to(device)
    
    # 定义BERT分类器，确保模型结构与训练时一致
    class BERTSentimentClassifier(nn.Module):
        def __init__(self, bert_model, hidden_size=768, num_labels=2, dropout_rate=0.3):
            super(BERTSentimentClassifier, self).__init__()
            self.bert = bert_model
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(hidden_size, num_labels)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits

    # 加载保存的模型
    model = BERTSentimentClassifier(bert_model).to(device)
    model.load_state_dict(torch.load('./model/BERT_sentiment_classifier_best.pth'))
    model.eval()  # 评估模式

    input_text = "我好开心啊"
    label = {1: "正面情绪", 0: "负面情绪"}

    # 预处理输入数据
    input_ids, attention_mask = preprocess_text(input_text, tokenizer)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    # 预测
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        predicted_class = torch.argmax(output, dim=1).item()

    predicted_label = label[predicted_class]
    print(f"预测文本: {input_text}")
    print(f"模型预测的类别: {predicted_label}")
