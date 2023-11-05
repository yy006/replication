import numpy as np
import pandas as pd
import torch
import preprocess
import trainer
import model
import data_preprocess


batch_size = 1024
path = 'dataset/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv'
preprocessor = data_preprocess.DataPreprocessor(path)
features_to_include = ['Src IP Addr', 'Dst IP Addr', 'Proto', 'Src Pt', 'Dst Pt', 'class', 'attackType', 'attackID', 'attackDescription']
processed_df = preprocessor.preprocess(num_rows=10, features=features_to_include)

X = processed_df.iloc[:, :5] # 文脈には5列だけ使う
d = X.to_numpy()
w2v,v2w = preprocess._w2v(d)
corpus = pd.DataFrame(preprocess._corpus(d, w2v)).to_numpy()
print(corpus)
freq  = preprocess._frequency(d)
print(freq)
train = preprocess._data_loader(corpus, batch_size)
print(train)

model = trainer.Trainer(w2v,v2w,freq,emb_dim=32)
model.fit(data = train,max_epoch=50,batch_size=256,neg_num=10)
torch.save(model.model.state_dict(),'ip2vec10.pth')