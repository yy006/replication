import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

# ファイルパスを適切に設定してください
dataset = "CIDDS-001" # CTU-13_Scenario
file_path = f'result/{dataset}/pth_model_w2v_v2w/model_w2v_v2w_10000.pth'

# ファイルをロード
saved_data = torch.load(file_path, map_location=torch.device('cpu'))

# Embeddingsを取得
embeddings = saved_data['model_state']['u_embedding.weight'].numpy()

# t-SNEで次元削減
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(embeddings)

# DBSCANでクラスタリング
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(tsne_results)

# v2w辞書を取得
v2w = saved_data['v2w']

# プロット用のデータフレームを作成
df = pd.DataFrame(tsne_results, columns=['x', 'y'])
df['label'] = [v2w[i] for i in range(len(tsne_results))]  # データポイントのワード
df['cluster'] = dbscan_labels

# インタラクティブなプロットを作成
fig = px.scatter(df, x='x', y='y', color='cluster', text='label',
                 hover_data=['label'], title='t-SNE and DBSCAN Clustering')

# プロットを表示
fig.show()
