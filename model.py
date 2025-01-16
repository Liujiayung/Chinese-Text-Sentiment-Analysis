import torch
from torch import nn
import matplotlib.pyplot as plt


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.5):
        super(GRUModel, self).__init__()        
        # 双向GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout_rate)        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 双向GRU输出的维度是hidden_size * 2       
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # GRU前向传播
        gru_out, _ = self.gru(x)        
        # 取序列的最后一个输出
        gru_out = gru_out[:, -1, :]       
        # Dropout
        gru_out = self.dropout(gru_out)      
        # 全连接层输出
        output = self.fc(gru_out)
        return output

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout_rate=0.5):
        super(LSTMModel, self).__init__()        
        # 双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout_rate)        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 双向LSTM输出的维度是hidden_size * 2       
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)        
        # 取序列的最后一个输出
        lstm_out = lstm_out[:, -1, :]       
        # Dropout
        lstm_out = self.dropout(lstm_out)      
        # 全连接层输出
        output = self.fc(lstm_out)
        return output

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_heads=4, dropout_rate=0.5):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Transformer Encoder层
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 输入嵌入
        x = self.embedding(x)
        # Transformer处理
        x = self.transformer_encoder(x)
        # 取序列的最后一个输出
        x = x[:, -1, :]
        # Dropout
        x = self.dropout(x)
        # 全连接层输出
        output = self.fc(x)
        return output

# 定义 CNN 模型
class CNNModel(nn.Module):
    def __init__(self, input_size, num_filters, filter_sizes, output_size):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, input_size)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_size)


    def forward(self, x):
        # 调整输入维度 [batch_size, 1, seq_length, input_size]
        x = x.unsqueeze(1)
        # 卷积操作
        conved = []
        for conv in self.conv_layers:
            padded_x = nn.functional.pad(x, (0, 0, 0, conv.kernel_size[0] - 1))
            conved.append(nn.functional.relu(conv(padded_x)).squeeze(3))
        # 最大池化
        pooled = [nn.functional.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in conved]
        # 拼接池化结果
        cat = torch.cat(pooled, dim=1)
        # 全连接层输出
        output = self.fc(cat)
        return output

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