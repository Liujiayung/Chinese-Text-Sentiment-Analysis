import torch
from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from data_process import load_tsv

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据读取
def load_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [line.strip() for line in f.readlines()]
    return data

train_texts = load_txt('train.txt')
test_texts = load_txt('test.txt')
_, train_labels = load_tsv("./data/train.tsv")
_, test_labels = load_tsv("./data/test.tsv")

# 加载BERT模型和分词器
bert_path = './bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_path)
bert_config = BertConfig.from_pretrained(bert_path)
bert_model = BertModel.from_pretrained(bert_path, ignore_mismatched_sizes=True).to(device)

# 文本编码函数
def encode_texts(texts, max_length=128):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return encoded['input_ids'], encoded['attention_mask']

# 将文本转换为张量
X_train_ids, X_train_masks = encode_texts(train_texts)
X_test_ids, X_test_masks = encode_texts(test_texts)

data_tensors = lambda ids, masks, labels: TensorDataset(ids, masks, torch.LongTensor(labels).to(device))
train_dataset = data_tensors(X_train_ids, X_train_masks, train_labels)
test_dataset = data_tensors(X_test_ids, X_test_masks, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义BERT分类器
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

# 加载最佳保存模型
model = BERTSentimentClassifier(bert_model).to(device)
model.load_state_dict(torch.load('./model/BERT_sentiment_classifier_best.pth'))
model.eval()

# 在测试集上进行评估
all_labels, all_preds = [], []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# 输出分类报告到文本文件
report = classification_report(all_labels, all_preds)
print(report)

# 将报告保存到文件
with open('bert_report.txt', 'w') as f:
    f.write(report)
