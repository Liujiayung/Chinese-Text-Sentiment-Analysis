import pandas as pd
import jieba
 
 
# 数据读取
def load_tsv(file_path):
    data = pd.read_csv(file_path, sep='\t')
    data_x = data.iloc[:, -1]
    data_y = data.iloc[:, 1]
    return data_x, data_y


train_x, train_y = load_tsv("./data/train.tsv")
test_x, test_y = load_tsv("./data/test.tsv")
dev_x, dev_y = load_tsv("./data/dev.tsv")
train_x=[list(jieba.cut(x)) for x in train_x]
test_x=[list(jieba.cut(x)) for x in test_x]
dev_x=[list(jieba.cut(x)) for x in dev_x]
 
with open('./hit_stopwords.txt','r',encoding='UTF8') as f:
    stop_words=[word.strip() for word in f.readlines()]
    print('Loading Stopwords Successfully')

def drop_stopword(datas):
    return [[word for word in data if word not in stop_words] for data in datas]

 
def save_data(datax,path):
    with open(path, 'w', encoding="UTF8") as f:
        for lines in datax:
            for i, line in enumerate(lines):
                f.write(str(line))
                # 如果不是最后一行，就添加一个逗号
                if i != len(lines) - 1:
                    f.write(',')
            f.write('\n')
 
if __name__ == '__main__':
    train_x=drop_stopword(train_x)
    test_x=drop_stopword(test_x)
    dev_x=drop_stopword(dev_x)
 
    save_data(train_x,'./train.txt')
    save_data(test_x,'./test.txt')
    save_data(dev_x,'./dev.txt')
    print('Processing data Successfully')