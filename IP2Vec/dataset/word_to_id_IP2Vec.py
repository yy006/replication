# coding: utf-8
import sys
import os
sys.path.append('..')
import pandas as pd
import pickle
import numpy as np

dataset_dir = os.path.dirname(os.path.abspath(__file__))

vocab_file = 'vcd_first1000rows_IP2Vec.vocab.pkl'

file_name = 'modified_botnet_first1000rows.csv'
file_path = dataset_dir + '/' + file_name

def load_vocab():
    vocab_path = dataset_dir + '/' + vocab_file

    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    df = pd.read_csv(file_path)
    # DataFrameのすべての要素を文字列に変換
    df = df.astype(str)

    # DataFrameのすべての要素を取得
    words = df.values.flatten()
    
    # ユニークな語彙のリストを作成
    unique_words = pd.unique(words)
    
    word_to_id = {}
    id_to_word = {}
    
    for i, word in enumerate(unique_words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word
    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word

def load_data():
    word_to_id, id_to_word = load_vocab()
    
    # CSVファイルからデータを読み込む
    df = pd.read_csv(file_path)

    # DataFrameのすべての要素を文字列に変換
    df = df.astype(str)
    
    # DataFrameの各要素をIDに変換
    # corpus = df.applymap(lambda x: word_to_id.get(x, -1))  # -1は該当しないワードの場合のデフォルト値
    # corpus = df.apply(lambda col: col.map(lambda x: word_to_id.get(x, -1)))
    # DataFrameをndarrayに変換
    # corpus = corpus.values
    arr = df.values

    # print(corpus)

    return arr, word_to_id