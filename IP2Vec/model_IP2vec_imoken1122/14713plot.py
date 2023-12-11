import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import data_preprocess
import re

# Src IP Addr列の要素でフィルタリングするためにdfを取得
path = 'dataset/CTU-13_Scenario-50/capture20110817_modified.csv'
preprocessor = data_preprocess.DataPreprocessor(path)
# features_to_include = ['Src IP Addr', 'Dst IP Addr', 'Proto', 'Src Pt', 'Dst Pt'] # classもついかしたい　
processed_df = preprocessor.preprocess_CTU13(num_rows=10000000) #start_date="2017-03-16 00:00:00"

# IPアドレスのリストを取得
ip_addresses = processed_df["Dst IP Addr"].unique()
# 147.32.X.Xの形のIPアドレスをフィルタリング
filtered_ip_addresses = [ip for ip in ip_addresses if re.match(r'^147\.32\.\d{1,3}\.\d{1,3}$', ip)]
print(filtered_ip_addresses)

# pthファイルパスの設定
# dataset = "CIDDS-001" 
dataset = "CTU-13_Scenario-50"
# file_path = f'result/{dataset}/pth_model_w2v_v2w/exweek4/model_w2v_v2w_exweek4_500000.pth'
file_path = f'result/{dataset}/pth_model_w2v_v2w/model_checkpoint_epoch_0.pth'
# ファイルをロード
saved_data = torch.load(file_path, map_location=torch.device('cpu'))

# v2w辞書とEmbeddingsを取得
v2w = saved_data['v2w']
embeddings = saved_data['model_state']['u_embedding.weight'].numpy()
print(embeddings.shape)

# フィルタリングのためのIPアドレスのリストを定義
ip_ranges = [f'147.32.{i}.{j}' for i in range(256) for j in range(256)]

'''
# v2w辞書から対応するインデックスを抽出
indices = [i for i, word in v2w.items() if word in ip_ranges]
filtered_embeddings = embeddings[indices]

# perplexityの値をフィルタリングされたサンプル数に合わせて調整
# perplexity_value = min(30, len(filtered_embeddings) - 1)  # 30 または サンプル数-1 の小さい方を使用

# t-SNEで次元削減
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1500)
tsne_results = tsne.fit_transform(filtered_embeddings)

# t-SNE結果をプロット
plt.figure(figsize=(12, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], marker='.')

# 各データポイントのIPアドレスをプロット
for i, index in enumerate(indices):
    plt.annotate(v2w[index], (tsne_results[i, 0], tsne_results[i, 1]), fontsize=8)

plt.title('Src IP Addresses - t-SNE Visualization')
plt.xlabel('t-SNE axis 1')
plt.ylabel('t-SNE axis 2')
plt.show()
'''
# ラベル情報を取得
labels = processed_df.groupby('Src IP Addr')['class'].apply(list).to_dict()

# v2w辞書から対応するインデックスとラベルを抽出
indices = []
colors = []
for i, word in v2w.items():
    if word in ip_ranges:
        indices.append(i)
        if word in labels:
            if 'Botnet' in labels[word]:
                colors.append('r')  # Botnetは赤色でプロット
            elif 'LEGITIMATE' in labels[word]:
                colors.append('b')  # LEGITIMATEは青色でプロット
            else:
                colors.append('g')  # それ以外は緑色でプロット
        else:
            colors.append('g')  # ラベルが存在しない場合も緑色でプロット

filtered_embeddings = embeddings[indices]

# t-SNEで次元削減
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1500)
tsne_results = tsne.fit_transform(filtered_embeddings)

# t-SNE結果をプロット
plt.figure(figsize=(12, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, marker='.')
plt.title('Src IP Addresses - t-SNE Visualization')
plt.xlabel('t-SNE axis 1')
plt.ylabel('t-SNE axis 2')
plt.show()