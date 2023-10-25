# coding: utf-8
import sys
import os
sys.path.append('..')

from dataset import word_to_id_IP2Vec

import numpy as np
import matplotlib.pyplot as plt
from preprocess import Preprocess
from model import ip2vec

flows, word_to_id = word_to_id_IP2Vec.load_data()

# データの準備
 # flows = [...]  # これはあなたのデータセットからのシーケンスのリストです。
# word_to_id = {...}  # これは単語をIDにマッピングする辞書です。
window_size = 5 # これはコンテキストのウィンドウサイズです。

# 前処理
preprocessor = Preprocess(flows, word_to_id, window_size)
X_train, y_train = preprocessor.generate_training_data()

# モデルの初期化
vocab_size = len(word_to_id)
emb_size = 100  # これは埋め込みのサイズです。適切な値に調整することができます。
model = ip2vec(vocab_size, emb_size)

# 学習
epochs = 5
learning_rate = 0.01
batch_size = 256
parameters = model.skipgram_model_training(X_train, y_train, vocab_size, emb_size, learning_rate, epochs, batch_size=batch_size, print_cost=True, plot_cost=True)
