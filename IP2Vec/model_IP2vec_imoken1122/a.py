'''
import torch
print(torch.__version__)
print(torch.cuda.is_available())
import torch
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.backends.cudnn.enabled)
'''
'''
import sklearn.cluster
'''
import numpy as np
import pandas as pd
import torch
import preprocess
import trainer
import model
import data_preprocess

import re

# 以下のコードは、選択した列で特定の条件を満たす行をフィルタリングし、それらの行を表示します。

# 正規表現パターン: 192.168.x.y で、yは0から16まで
pattern = r'^192\.168\.\d+\.(1[0-6]|[0-9])$'

batch_size = 1024
path = 'dataset/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv'
preprocessor = data_preprocess.DataPreprocessor(path)
features_to_include = ['Src IP Addr', 'Dst IP Addr', 'Proto', 'Src Pt', 'Dst Pt', 'class', 'attackType', 'attackID', 'attackDescription']
processed_df = preprocessor.preprocess(num_rows=10000000, features=features_to_include) #start_date="2017-03-16 00:00:00"

# 'Src IP Addr' 列で '192.168' を含む行のみをフィルタリング
filtered_df = processed_df[processed_df['Src IP Addr'].str.contains(pattern, regex=True)]

# 各ユニークな 'Src IP Addr' の出現回数をカウント
ip_counts = filtered_df['Src IP Addr'].value_counts()

# 結果を表示
print(ip_counts)
