import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ファイルパスを適切に設定してください
dataset = "CIDDS-001" # CTU-13_Scenario
file_path = f'result/{dataset}/pth_model_w2v_v2w/model_w2v_v2w_500000_0316_epoch10'

# ファイルをロード
saved_data = torch.load(file_path, map_location=torch.device('cpu'))

# Embeddingsを取得
embeddings = saved_data['model_state']['u_embedding.weight'].numpy()

# t-SNEで次元削減
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(embeddings)
'''
# DBSCANでクラスタリング
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(tsne_results)

# クラスタリングの結果を確認
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f'Identified clusters: {n_clusters}')
print(f'Noise points: {n_noise}')
'''
# t-SNE結果をプロット
plt.figure(figsize=(12, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1],  cmap='viridis', marker='.') #c=dbscan_labels,
'''
# データポイントのインデックスまたはラベルを表示（サンプルサイズが小さい場合）
for i, txt in enumerate(range(len(tsne_results))):
    plt.annotate(txt, (tsne_results[i, 0], tsne_results[i, 1]), fontsize=8)
'''
plt.title('t-SNE and DBSCAN Clustering')
plt.xlabel('t-SNE axis 1')
plt.ylabel('t-SNE axis 2')
plt.colorbar()
plt.show()
