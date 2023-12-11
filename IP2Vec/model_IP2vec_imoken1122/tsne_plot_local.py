import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import data_preprocess

# Src IP Addr列の要素でフィルタリングするためにdfを取得
path = 'dataset/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv'
preprocessor = data_preprocess.DataPreprocessor(path)
features_to_include = ['Src IP Addr', 'Dst IP Addr', 'Proto', 'Src Pt', 'Dst Pt', 'class', 'attackType', 'attackID', 'attackDescription']
processed_df = preprocessor.preprocess(num_rows=10000000) #start_date="2017-03-16 00:00:00"


# pthファイルパスの設定
dataset = "CIDDS-001" # CTU-13_Scenario
file_path = f'result/{dataset}/pth_model_w2v_v2w/opweek1/model_w2v_v2w_500000_0316_epoch10.pth'

# ファイルをロード
saved_data = torch.load(file_path, map_location=torch.device('cpu'))
'''
#全データを次元削減した後に、192.168を含み、Src IP Addr列にも含まれるもののみをプロットする
# v2w辞書とEmbeddingsを取得
v2w = saved_data['v2w']
embeddings = saved_data['model_state']['u_embedding.weight'].numpy()

# Src IP Addr列に出現するユニークなIPアドレスを取得
unique_ips = processed_df['Src IP Addr'].unique()

# すべてのデータでt-SNEで次元削減
perplexity_value = min(30, len(embeddings) - 1)
tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity_value, n_iter=300)
tsne_results = tsne.fit_transform(embeddings)

# 192.168を含み、Src IP Addr列にも含まれるインデックスを抽出
indices_to_plot = [i for i, word in v2w.items() if(word in unique_ips)and(isinstance(word, str) and word.startswith('192.168'))]

# t-SNE結果をプロット
plt.figure(figsize=(12, 8))

# 192.168を含み、Src IP Addr列にも含まれるデータポイントのみプロット
for i in indices_to_plot:
    plt.scatter(tsne_results[i, 0], tsne_results[i, 1], marker='.')
    plt.annotate(v2w[i], (tsne_results[i, 0], tsne_results[i, 1]), fontsize=8)

plt.title('Src IP Addresses - t-SNE Visualization')
plt.xlabel('t-SNE axis 1')
plt.ylabel('t-SNE axis 2')
plt.show()
'''

'''
# 192.168を含み、Src IP Addr列にも含まれるデータのみを次元削減
# v2w辞書とEmbeddingsを取得
v2w = saved_data['v2w']
embeddings = saved_data['model_state']['u_embedding.weight'].numpy()

# Src IP Addr列に出現するユニークなIPアドレスを取得
unique_ips = processed_df['Src IP Addr'].unique()

# v2w辞書から対応するインデックスを抽出
# 192.168で始まるIPアドレスに対応するインデックスを抽出
indices = [i for i, word in v2w.items() if(word in unique_ips)and(isinstance(word, str) and word.startswith('192.168'))]
filtered_embeddings = embeddings[indices]

# perplexityの値をフィルタリングされたサンプル数に合わせて調整
perplexity_value = min(30, len(filtered_embeddings) - 1)

# t-SNEで次元削減
tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity_value, n_iter=300)
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

# v2w辞書とEmbeddingsを取得
v2w = saved_data['v2w']
embeddings = saved_data['model_state']['u_embedding.weight'].numpy()

# フィルタリングのためのIPアドレスのリストを定義
ip_ranges = [
    '192.168.100.3', '192.168.100.4', '192.168.100.5', '192.168.100.6',
    '192.168.200.3', '192.168.200.4', '192.168.200.5', '192.168.200.8', '192.168.200.9',
    '192.168.210.3', '192.168.210.4', '192.168.210.5'
]
ip_ranges += [f'192.168.220.{i}' for i in range(3, 17)]

# v2w辞書から対応するインデックスを抽出
indices = [i for i, word in v2w.items() if word in ip_ranges]
filtered_embeddings = embeddings[indices]

# perplexityの値をフィルタリングされたサンプル数に合わせて調整
# perplexity_value = min(30, len(filtered_embeddings) - 1)  # 30 または サンプル数-1 の小さい方を使用

'''
# t-SNEで次元削減
tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity_value, n_iter=300)
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
# 192.168を削除
# t-SNEで次元削減
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1500)
tsne_results = tsne.fit_transform(filtered_embeddings)

# t-SNE結果をプロット
plt.figure(figsize=(12, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], marker='.')

# 各データポイントの名前（'192.168.' を除いた2バイト分）をプロット
for i, index in enumerate(indices):
    # IPアドレスから '192.168.' を取り除いて2バイト分のみを取得
    ip_label = v2w[index].replace('192.168.', '')
    ip_label = '.'.join(ip_label.split('.')[:2])  # 最初の2バイトのみを取得

    plt.annotate(ip_label, (tsne_results[i, 0], tsne_results[i, 1]), fontsize=8)

plt.title('192.168 Starting IP Addresses - t-SNE Visualization')
plt.xlabel('t-SNE axis 1')
plt.ylabel('t-SNE axis 2')
plt.show()
