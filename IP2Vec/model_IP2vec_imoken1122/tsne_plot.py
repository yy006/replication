import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

# ファイルをロード
# dataset = "CIDDS-001" 
dataset = "CTU-13_Scenario-50"
# file_path = f'result/{dataset}/pth_model_w2v_v2w/exweek4/model_w2v_v2w_exweek4_500000.pth'
file_path = f'result/{dataset}/model_checkpoint_epoch_0.pth'

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
# dbscan = DBSCAN(eps=0.5, min_samples=5)
# dbscan_labels = dbscan.fit_predict(tsne_results)

# プロット用のデータフレームを作成
df = pd.DataFrame(tsne_results, columns=['x', 'y'])
df['word'] = filtered_words  # フィルタリングされたワードをデータフレームに追加
# df['cluster'] = dbscan_labels

# インタラクティブなプロットを作成
#fig = px.scatter(df, x='x', y='y', color='cluster', text='word',
#                 hover_data=['word'], title='Filtered IP Address t-SNE and DBSCAN Clustering')

fig = px.scatter(df, x='x', y='y', color='word',
                 hover_data=['word'], title='Filtered IP Address t-SNE')

# 特定のワードリストを定義
#attacker = ['ATTACKER1', 'ATTACKER2']
attacker = ['ATTACKER3']
normal = ['EXT_SERVER', 'OPENSTACK_NET']
unknown = ['32158_172', '32183_156', '27561_114', '30921_152']
suspicious = ['10006_27', '29376_114', '17800_126', '32955_29', '14105_26']

# 各データポイントにカテゴリを割り当てる関数
def assign_category(word):
    if word in attacker:
        return 'attacker'
    elif word in normal:
        return 'normal'
    elif word in unknown:
        return 'unknown'
    elif word in suspicious:
        return 'suspicious'
    else:
        return 'Other'

# カテゴリカラムを追加
df['category'] = df['word'].apply(assign_category)

# インタラクティブなプロットを作成（'category' カラムに基づいて色を設定）
fig = px.scatter(df, x='x', y='y', color='category',
                 hover_data=['word'])


# プロットを表示
fig.show()