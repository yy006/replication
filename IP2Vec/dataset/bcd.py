# coding: utf-8
import sys
import os
sys.path.append('..')
try:
    import urllib.request
except ImportError:
    raise ImportError('Use Python3!')
import pickle
import numpy as np

vocab_file = 'ptb.vocab.pkl'

dataset_dir = os.path.dirname(os.path.abspath(__file__))

def load_vocab():
    # 語彙ファイルの保存先パス生成
    vocab_path = dataset_dir + '/' + vocab_file 

    # 語彙ファイルの存在チェック
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    # 語彙マッピングの初期化
    word_to_id = {}
    id_to_word = {}

    # トレーニングデータの読み込み(ダウンロード)(変える必要あり)
    data_type = 'train'
    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name

    _download(file_name)

    # 単語リストの生成(変える必要あり)
    words = open(file_path).read().replace('\n', '<eos>').strip().split()

    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word

