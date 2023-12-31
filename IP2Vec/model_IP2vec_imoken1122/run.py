import numpy as np
import pandas as pd
import torch
import preprocess
import trainer
import model
import data_preprocess

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

batch_size = 1024
# path = 'dataset/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv'
path = 'dataset/CTU-13_Scenario-50/capture20110817_modified.csv'
preprocessor = data_preprocess.DataPreprocessor(path)
'''
features_to_include = ['Src IP Addr', 'Dst IP Addr', 'Proto', 'Src Pt', 'Dst Pt', 'class', 'attackType', 'attackID', 'attackDescription']
processed_df = preprocessor.preprocess(num_rows=10000000, features=features_to_include) #start_date="2017-03-16 00:00:00"
'''
processed_df = preprocessor.preprocess_CTU13(num_rows=100000000) #start_date="2017-03-16 00:00:00"
processed_df = preprocessor.preprocess_CTU13(num_rows=100000000) #start_date="2017-03-16 00:00:00"

X = processed_df.iloc[:, :5] # 文脈には5列だけ使う
print(X)
d = X.to_numpy()
w2v,v2w = preprocess._w2v(d)
corpus = pd.DataFrame(preprocess._corpus(d, w2v)).to_numpy()
#print(corpus)
freq  = preprocess._frequency(d)
#print(freq)
train = preprocess._data_loader(corpus, batch_size)
#print(train)

model = trainer.Trainer(w2v,v2w,freq,emb_dim=32)
model.fit(data = train,max_epoch=50,batch_size=256,neg_num=10, patience_limit=3)

# モデルの状態辞書を取得
model_state = model.model.state_dict()

# w2vとv2wを辞書に追加
save_dict = {
    'model_state': model_state,
    'w2v': w2v,
    'v2w': v2w
}

# 辞書を.pthファイルとして保存
torch.save(save_dict, 'model_w2v_v2w_CTU13_all.pth')
