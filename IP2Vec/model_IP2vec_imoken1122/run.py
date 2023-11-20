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
path = 'dataset/CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week4.csv'
preprocessor = data_preprocess.DataPreprocessor(path)
features_to_include = ['Src IP Addr', 'Dst IP Addr', 'Proto', 'Src Pt', 'Dst Pt', 'class', 'attackType', 'attackID', 'attackDescription']
processed_df = preprocessor.preprocess(num_rows=500000, features=features_to_include) #start_date="2017-03-16 00:00:00"

X = processed_df.iloc[:, :5] # 文脈には5列だけ使う
d = X.to_numpy()
w2v,v2w = preprocess._w2v(d)
corpus = pd.DataFrame(preprocess._corpus(d, w2v)).to_numpy()
#print(corpus)
freq  = preprocess._frequency(d)
#print(freq)
train = preprocess._data_loader(corpus, batch_size)
#print(train)

model = trainer.Trainer(w2v,v2w,freq,emb_dim=32)
model.fit(data = train,max_epoch=50,batch_size=256,neg_num=10, patience_limit=4)

# モデルの状態辞書を取得
model_state = model.model.state_dict()

# w2vとv2wを辞書に追加
save_dict = {
    'model_state': model_state,
    'w2v': w2v,
    'v2w': v2w
}

# 辞書を.pthファイルとして保存
torch.save(save_dict, 'model_w2v_v2w_exweek4_500000.pth')
