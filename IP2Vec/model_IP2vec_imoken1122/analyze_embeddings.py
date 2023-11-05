from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import torch
import preprocess
import trainer
import model
import data_preprocess

# .pthファイルから辞書をロード
loaded_dict = torch.load('result/CIDDS-001/model_w2v_v2w_10.pth')

# モデルの状態辞書を取得
model_state = loaded_dict['model_state']

# w2vとv2wを取得
w2v = loaded_dict['w2v']
v2w = loaded_dict['v2w']

# 1. 学習された分散表現を取得
# embeddings = model.u_embedding.weight.cpu().detach().numpy() gpuからcpuに移して...
embeddings = model_state['u_embedding.weight'].cpu().detach().numpy()

# 2. t-SNEを使用して次元削減
tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = tsne.fit_transform(embeddings)

# 3. DBScanを使用してクラスタリング
clustering = DBSCAN(eps=0.5, min_samples=5).fit(embeddings_2d)
labels = clustering.labels_

# 4. matplotlibを使用して可視化
plt.figure(figsize=(10, 10))
for i, label in enumerate(labels):
    x, y = embeddings_2d[i]
    plt.scatter(x, y, color=plt.cm.jet(float(label) / np.max(labels + 1)))
    plt.annotate(v2w[i], xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=8)

plt.title('t-SNE visualization of word embeddings with DBScan clustering')
plt.show()
