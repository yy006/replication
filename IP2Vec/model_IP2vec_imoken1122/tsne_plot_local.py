import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ファイルパスを適切に設定してください
dataset = "CIDDS-001" # CTU-13_Scenario
file_path = f'result/{dataset}/pth_model_w2v_v2w/opweek1/model_checkpoint_epoch_0.pth'

# ファイルをロード
saved_data = torch.load(file_path, map_location=torch.device('cpu'))

# v2w辞書とEmbeddingsを取得
v2w = saved_data['v2w']
embeddings = saved_data['model_state']['u_embedding.weight'].numpy()

# 192.168で始まるIPアドレスに対応するインデックスを抽出
indices = [i for i, word in v2w.items() if isinstance(word, str) and word.startswith('192.168')]
filtered_embeddings = embeddings[indices]

# perplexityの値をフィルタリングされたサンプル数に合わせて調整
perplexity_value = min(30, len(filtered_embeddings) - 1)  # 30 または サンプル数-1 の小さい方を使用

# その他のコード...

# t-SNEで次元削減
tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity_value, n_iter=300)
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
