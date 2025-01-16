import pandas as pd
import torch
from torch import nn
import jieba
from gensim.models import Word2Vec
import numpy as np
from data_process import load_tsv
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt  

# 检查是否有可用的 GPU，如果有则使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据读取
def load_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [[line.strip()] for line in f.readlines()]
        return data

train_x = load_txt('train.txt')
test_x = load_txt('test.txt')
train = train_x + test_x
X_all = [i for x in train for i in x]

_, train_y = load_tsv("./data/train.tsv")
_, test_y = load_tsv("./data/test.tsv")

# 训练 Word2Vec 模型
word2vec_model = Word2Vec(sentences=X_all, vector_size=100, window=5, min_count=1, workers=4)

# 将文本转换为 Word2Vec 向量表示
def text_to_vector(text):
    vector = [word2vec_model.wv[word] for word in text if word in word2vec_model.wv]
    return sum(vector) / len(vector) if vector else [0] * word2vec_model.vector_size

X_train_w2v = [[text_to_vector(text)] for line in train_x for text in line]
X_test_w2v = [[text_to_vector(text)] for line in test_x for text in line]


# 将词向量转换为 PyTorch 张量
X_train_array = np.array(X_train_w2v, dtype=np.float32)
X_train_tensor = torch.Tensor(X_train_array).to(device)  # 将数据移到 GPU
X_test_array = np.array(X_test_w2v, dtype=np.float32)
X_test_tensor = torch.Tensor(X_test_array).to(device)  # 将数据移到 GPU

# 使用 DataLoader 打包文件
train_dataset = TensorDataset(X_train_tensor, torch.LongTensor(train_y).to(device))  # 将标签移到 GPU
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, torch.LongTensor(test_y).to(device))  # 将标签移到 GPU
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead=4, num_layers=2, dim_feedforward=256):
        super(TransformerModel, self).__init__()
        # 位置编码
        self.pos_encoder = PositionalEncoding(input_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, x):
        # 调整输入形状：[batch_size, seq_length, input_size]
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        # 取最后一个位置的输出
        output = output[-1, :, :]
        output = self.linear(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 定义模型
input_size = word2vec_model.vector_size
output_size = 2  # 输出的大小，根据你的任务而定

model = TransformerModel(input_size, output_size, d_model=input_size).to(device)  # 将模型移到 GPU

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
# 使用 L2 正则化（weight_decay）
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)  # L2 正则化项


if __name__ == "__main__":
    # 记录训练损失的列表
    train_losses = []


    # 训练模型
    num_epochs = 50
    log_interval = 100  # 每隔 100 个批次输出一次日志
    loss_min = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0  # 每个 epoch 的累计损失
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)  # 将数据和标签移到 GPU
            outputs = model(data)
            loss = criterion(outputs, target)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            epoch_loss += loss.item()


            if batch_idx % log_interval == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, batch_idx, len(train_loader), loss.item()))


            # 保存最佳模型
            if loss.item() < loss_min:
                loss_min = loss.item()
                torch.save(model, './model/Transformer_model.pth')


        # 计算并记录每个 epoch 的平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_epoch_loss:.4f}")


    # 绘制训练损失变化图
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Transformer Training Loss Curve')
    plt.legend()
    # 保存图像到当前目录
    plt.savefig('Transformer_train_loss_plot.png')


    plt.show()


    # 模型评估
    with torch.no_grad():
        model.eval()
        all_labels = []
        all_preds = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(target.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())


        print(classification_report(all_labels, all_preds))