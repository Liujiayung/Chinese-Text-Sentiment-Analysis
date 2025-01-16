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

# 对抗训练：FGM
class FGM:
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# 初始化模型
model = BERTSentimentClassifier(bert_model).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()  # 混合精度
fgm = FGM(model)  # 初始化对抗训练

if __name__ == "__main__":
    train_losses = []
    num_epochs = 5
    log_interval = 10

    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast():  # 混合精度训练
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()  # 缩放梯度

            # 对抗训练（FGM）
            fgm.attack()  # 添加对抗扰动
            with torch.cuda.amp.autocast():
                outputs_adv = model(input_ids, attention_mask)
                loss_adv = criterion(outputs_adv, labels)
            scaler.scale(loss_adv).backward()  # 对抗梯度
            fgm.restore()  # 恢复原始权重

            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放因子

            epoch_loss += loss.item()
            if batch_idx % log_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_epoch_loss:.4f}")

        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), './model/BERT_sentiment_classifier_best.pth')

    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BERT Training Loss Curve')
    plt.legend()
    plt.savefig('BERT_train_loss_plot.png')
    plt.show()

    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print(classification_report(all_labels, all_preds))
