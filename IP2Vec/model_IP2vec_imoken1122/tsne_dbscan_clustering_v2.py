import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

# ファイルをロード
dataset = "CIDDS-001" # CTU-13_Scenario
file_path = f'result/{dataset}/pth_model_w2v_v2w/exweek3/model_w2v_v2w_exweek3_500000.pth'

saved_data = torch.load(file_path, map_location=torch.device('cpu'))

# v2w辞書を取得
v2w = saved_data['v2w']

# Embeddingsを取得
embeddings = saved_data['model_state']['u_embedding.weight'].numpy()

# 文字列型のワードと対応するベクトルのみを選択
filtered_indices = [index for index, word in v2w.items() if isinstance(word, str)]
filtered_embeddings = embeddings[filtered_indices]
filtered_words = [v2w[i] for i in filtered_indices]

# t-SNEで次元削減
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(filtered_embeddings)

# DBSCANでクラスタリング
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(tsne_results)

# プロット用のデータフレームを作成
df = pd.DataFrame(tsne_results, columns=['x', 'y'])
df['word'] = filtered_words  # フィルタリングされたワードをデータフレームに追加
df['cluster'] = dbscan_labels

# インタラクティブなプロットを作成
#fig = px.scatter(df, x='x', y='y', color='cluster', text='word',
#                 hover_data=['word'], title='Filtered IP Address t-SNE and DBSCAN Clustering')

fig = px.scatter(df, x='x', y='y', color='cluster',
                 hover_data=['word'], title='Filtered IP Address t-SNE and DBSCAN Clustering')
# 強調表示するワードのリスト
highlight_words = ['EXT_SERVER', 'ATTACKER2', 'ATTACKER1']

# 強調表示するワードをデータフレームから見つける
highlight_df = df[df['word'].isin(highlight_words)]

# これらのポイントを強調してプロットに追加
for word, row in highlight_df.iterrows():
    fig.add_trace(go.Scatter(x=[row['x']], y=[row['y']],
                             text=[row['word']],
                             mode='markers+text',
                             textposition='top center',
                             marker=dict(size=10, color='LightSkyBlue', line=dict(width=2, color='DarkSlateGrey')),
                             showlegend=False))

# プロットを表示
fig.show()