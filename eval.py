import torch
from sklearn.metrics import classification_report
from data_process import load_tsv
import numpy as np
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, TensorDataset
from model import GRUModel,LSTMModel,TransformerModel,CNNModel

# 定义评估函数
def evaluate_model(model_path, test_loader, device):
    model = torch.load(model_path)  # 加载模型
    model.to(device)  # 将模型移到GPU（如果有）

    # 模型评估
    model.eval()  # 设置模型为评估模式
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)  # 预测输出
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别

            all_labels.extend(target.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 输出分类报告
    return classification_report(all_labels, all_preds)

# 假设您的测试集和设备已经准备好
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据读取
def load_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [[line.strip()] for line in f.readlines()]
        return data
test_x = load_txt('test.txt')
test = test_x
X_all = [i for x in test for i in x]
# 训练Word2Vec模型
word2vec_model = Word2Vec(sentences=X_all, vector_size=100, window=5, min_count=1, workers=4)

# 将文本转换为Word2Vec向量表示
def text_to_vector(text):
    vector = [word2vec_model.wv[word] for word in text if word in word2vec_model.wv]
    return sum(vector) / len(vector) if vector else [0] * word2vec_model.vector_size

X_test_w2v = [text_to_vector(text) for line in test_x for text in line]
# 读取测试集
_, test_y = load_tsv("./data/test.tsv")
X_test_array = np.array(X_test_w2v, dtype=np.float32)
X_test_tensor = torch.Tensor(X_test_array).unsqueeze(1).to(device)  # 添加维度
# 定义 TensorDataset
test_dataset = TensorDataset(X_test_tensor, torch.LongTensor(test_y).to(device))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 评估时不需

# 对每个模型进行评估
cnn_report = evaluate_model('./model/CNN_model.pth', test_loader, device)
lstm_report = evaluate_model('./model/LSTM_model.pth', test_loader, device)
gru_report = evaluate_model('./model/GRU_model.pth', test_loader, device)

# 打印或保存分类报告
with open("model_comparison_report.txt", "w", encoding="utf-8") as f:
    f.write("CNN Model Classification Report:\n")
    f.write(cnn_report + "\n\n")  # 写入CNN模型的报告
    f.write("LSTM Model Classification Report:\n")
    f.write(lstm_report + "\n\n")  # 写入LSTM模型的报告
    f.write("GRU Model Classification Report:\n")
    f.write(gru_report + "\n\n")  # 写入GRU模型的报告

print("模型分类报告已保存到 model_comparison_report.txt 文件中。")
